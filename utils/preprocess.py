import torch
import json
import pickle as pkl
import numpy as np
from Transformer_Explanation.modules.BERT.BertForTask import BertForSequenceClassification, BertForQuestionAnswering
from Transformer_Explanation.modules.BERT.RobertaForTask import RobertaForQuestionAnswering
from Transformer_Explanation.modules.GPT.GPT_model import GPT2ForQuestionAnswering
from Transformer_Explanation.modules.LLaMA.LLaMA_model import LlamaForSequenceClassification
from transformers import AutoTokenizer

task_to_keys = {
    "imdb": ("text", None),
    "yelp": ("text", None),
    "mnli": ("premise", "hypothesis"),
    "qqp": ("question1", "question2"),
    "sst2": ("sentence", None),
    "squadv1": ("context", "question"),
    "squadv2": ("context", "question")
}
model_keys = {
    'bert': BertForQuestionAnswering,
    'roberta': RobertaForQuestionAnswering,
    'gpt2': GPT2ForQuestionAnswering,
    'llama': LlamaForSequenceClassification
}


def _head_mask(type, **kwargs):
    if type == 'None':
        return None
    
    if type == 'indices':
        mask_indices = kwargs['mask_indices']
        mask = np.zeros((kwargs['num_layers'], kwargs['num_heads']))
        for m in mask_indices:
            mask[m[0], m[1]] = 1
        return mask
    
    if type == 'indices_inv':
        mask_indices = kwargs['mask_indices']
        mask = np.ones((kwargs['num_layers'], kwargs['num_heads']))
        for m in mask_indices:
            mask[m[0], m[1]] = 0
        return mask

    if type == 'random':
        mask = np.zeros(kwargs['num_layers'] * kwargs['num_heads'], dtype=int)

        forbidden_indices = []
        if kwargs.get('preserve_pos') is not None:
            forbidden_indices = [pos[0] * kwargs['num_heads'] + pos[1] for pos in kwargs['preserve_pos']]
        
        available_positions = set(range(kwargs['num_layers'] * kwargs['num_heads'])) - set(forbidden_indices)

        if kwargs.get('num_ones') is not None:
            num_ones = kwargs['num_ones']
        else:
            num_ones = kwargs['num_layers'] * kwargs['num_heads']

        ones_positions = np.random.choice(list(available_positions), size=num_ones, replace=False)
        mask[ones_positions] = 1

        return mask.reshape(kwargs['num_layers'], kwargs['num_heads'])
    

def load_dataset_json(type = 'train', dataset = None, data_path = None):
    data = []
    dataset = dataset
    with open(f'{data_path}/{dataset}/{dataset}_parsed_{type}.json', 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            data.append(json.loads(line))
    return data


def load_model_and_tokenizer(name, device, is_qa=False):
    device = torch.device(device)
    if is_qa:
        model_name = name.split("/")[-1].split("-")[0]
        model = model_keys[model_name].from_pretrained(name).to(device)
    else:
        model = BertForSequenceClassification.from_pretrained(name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer


def load_model(name, device, is_qa=False):
    device = torch.device(device)
    if is_qa:
        return BertForQuestionAnswering.from_pretrained(name).to(device)
    else:
        return BertForSequenceClassification.from_pretrained(name).to(device)


def load_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)


def preprocess_sample(text, tokenizer, device, dataset):
    special_tokens = {'[SEP]'}
    special_idxs = {102}

    if dataset in ['qqp', 'mnli']:
        sentence1_key, sentence2_key = task_to_keys[dataset]
        args = (
            (text[sentence1_key],) if sentence2_key is None else (text[sentence1_key], text[sentence2_key])
        )
        tokenized_input = tokenizer(*args, padding=False, max_length=tokenizer.model_max_length, truncation=True)

    elif dataset in ['squadv1', 'squadv2']:
        ans_idx = 'answer' if dataset == 'squadv1' else 'answers'
        inputs = tokenizer(
            text['question'].strip(),
            text["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding=False,
        )
        offset_mapping = inputs.pop("offset_mapping")
        answers = text[ans_idx]
        start_positions = []
        end_positions = []

        for offset in offset_mapping:
            start_char = answers["answer_start"][0]
            end_char = answers["answer_start"][0] + len(answers["text"][0])
            sequence_ids = inputs.sequence_ids(0)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1 and idx < (len(sequence_ids) - 1):
                idx += 1
            context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset_mapping[context_start][0] > end_char or offset_mapping[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset_mapping[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset_mapping[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions[0]
        inputs["end_positions"] = end_positions[0]

        input_ids = inputs['input_ids']
        text_ids = (torch.tensor([input_ids])).to(device)
        token_words = tokenizer.convert_ids_to_tokens(text_ids[0])
        
        att_mask = inputs['attention_mask']
        special_idxs = [x for x, y in list(enumerate(input_ids)) if y in special_tokens]
        att_mask = [0 if index in special_idxs else 1 for index, item in enumerate(att_mask)]
        att_mask = (torch.tensor([att_mask])).to(device)
        
        return text_ids, att_mask, token_words, start_positions, end_positions
    else:
        tokenized_input  = tokenizer(text, add_special_tokens=True, truncation=True)
    input_ids = tokenized_input['input_ids']
    token_type_ids = torch.tensor(tokenized_input['token_type_ids']).to(device)
    text_ids = (torch.tensor([input_ids])).to(device)
    text_words = tokenizer.convert_ids_to_tokens(text_ids[0])
    
    # mask special tokens
    att_mask = tokenized_input['attention_mask']
    spe_idxs = [x for x, y in list(enumerate(input_ids)) if y in special_idxs]
    att_mask = [0 if index in spe_idxs else 1 for index, item in enumerate(att_mask)]
    att_mask = (torch.tensor([att_mask])).to(device)
    
    return text_ids, att_mask, text_words, token_type_ids


def generate_head_mask(parsed_path, 
                       dataset, 
                       model_name, 
                       mask_type, 
                       synt_acc_threshold, pos_threshold, 
                       num_layers, num_heads, 
                       corruption_rate=0):
    print('loading deprel res from cache...')
    if model_name != '':
        model_name = '_' + model_name

    with open(f'{parsed_path}{dataset}{model_name}_total_deprel_hit_res.pkl', 'rb') as f:
        deprel_hit = pkl.load(f)

    with open(f'{parsed_path}{dataset}{model_name}_total_deprel_miss_res.pkl', 'rb') as f:
        deprel_miss = pkl.load(f)

    with open(f'{parsed_path}{dataset}{model_name}_rel_ppos_res.pkl', 'rb') as f:
        rel_ppos_res = pkl.load(f)

    with open(f'{parsed_path}{dataset}{model_name}_rel_npos_res.pkl', 'rb') as f:
        rel_npos_res = pkl.load(f)

    # syntactic
    deprel_total = deprel_hit + deprel_miss
    deprel_acc = deprel_hit / deprel_total

    # positional
    total_pos_stats = (rel_ppos_res + rel_npos_res).sum(axis=2)[0][0]
    ppos_freq = rel_ppos_res / total_pos_stats
    npos_freq = rel_npos_res / total_pos_stats

    # syntactic mask
    synt_mask_nsubj = np.argwhere(deprel_acc > synt_acc_threshold[0])[np.where(np.argwhere(deprel_acc > synt_acc_threshold[0])[:, 2] == 0)]
    synt_mask_obj = np.argwhere(deprel_acc > synt_acc_threshold[1])[np.where(np.argwhere(deprel_acc > synt_acc_threshold[1])[:, 2] == 1)]
    synt_mask_amod = np.argwhere(deprel_acc > synt_acc_threshold[2])[np.where(np.argwhere(deprel_acc > synt_acc_threshold[2])[:, 2] == 2)]
    synt_mask_advmod = np.argwhere(deprel_acc > synt_acc_threshold[3])[np.where(np.argwhere(deprel_acc > synt_acc_threshold[3])[:, 2] == 3)]

    # pos mask
    pos_mask = np.vstack([np.argwhere(ppos_freq > pos_threshold), np.argwhere(npos_freq > pos_threshold)])


    gen_type = 'indices'
    if mask_type == 'synt':
        mask = np.vstack([synt_mask_nsubj, synt_mask_obj, synt_mask_amod, synt_mask_advmod])

    elif mask_type == 'pos':
        mask = np.vstack([pos_mask])

    elif mask_type == 'synt_pos':
        mask = np.vstack([synt_mask_nsubj, synt_mask_obj, synt_mask_amod, synt_mask_advmod, pos_mask])

    elif mask_type == 'synt_pos_corruption':
        mask = np.vstack([synt_mask_nsubj, synt_mask_obj, synt_mask_amod, synt_mask_advmod, pos_mask])
        preserve_pos = mask[:, :2]
        num_ones = int((num_layers * num_heads - preserve_pos.shape[0]) * corruption_rate)
        corruption_mask = torch.tensor(_head_mask(type='random', num_layers=num_layers, num_heads=num_heads, 
                                                  num_ones=num_ones, preserve_pos=preserve_pos))
    elif mask_type == 'random_abla':
        mask = np.vstack([synt_mask_nsubj, synt_mask_obj, synt_mask_amod, synt_mask_advmod, pos_mask])
        num_ones = mask.shape[0]
        corruption_mask = torch.tensor(_head_mask(type='random', num_layers=num_layers, num_heads=num_heads, 
                                                  num_ones=num_ones))
    elif mask_type == 'random':
        gen_type = 'random'
        mask = None

    head_mask = torch.tensor(_head_mask(type=gen_type, mask_indices=mask, num_layers=num_layers, num_heads=num_heads))

    if mask_type == 'synt_pos_corruption':
        head_mask = head_mask + corruption_mask

    elif mask_type == 'random_abla':
        head_mask = corruption_mask
    
    return head_mask


