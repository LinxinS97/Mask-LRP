import torch
import json
import numpy as np
from BERT_explainability.modules.BERT.BertForTask import BertForSequenceClassification, BertForQuestionAnswering
from transformers import AutoTokenizer


### FOR YELP, SST2
task_to_keys = {
    "cola": ("sentence", None),
    "yelp": ("text", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def generate_head_mask(type, **kwargs):
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
        return np.random.randint(0, 2, (kwargs['num_layers'], kwargs['num_heads']))
    

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
        model = BertForQuestionAnswering.from_pretrained(name).to(device)
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
    special_tokens = {'[SEP]', '[CLS]'}
    special_idxs = {101, 102}
    if dataset in ['qqp', 'mnli']:
        sentence1_key, sentence2_key = task_to_keys[dataset]
        args = (
            (text[sentence1_key],) if sentence2_key is None else (text[sentence1_key], text[sentence2_key])
        )
        tokenized_input = tokenizer(*args, padding=False, max_length=tokenizer.model_max_length, truncation=True)

    elif dataset in ['squadv1', 'squadv2']:
        inputs = tokenizer(
            text['question'].strip(),
            text["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding=False,
        )
        offset_mapping = inputs.pop("offset_mapping")
        answers = text["answer"]
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
            while sequence_ids[idx] == 1:
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
        text_words = tokenizer.convert_ids_to_tokens(text_ids[0])
        
        att_mask = inputs['attention_mask']
        special_idxs = [x for x, y in list(enumerate(input_ids)) if y in special_tokens]
        att_mask = [0 if index in special_idxs else 1 for index, item in enumerate(att_mask)]
        att_mask = (torch.tensor([att_mask])).to(device)
        
        return text_ids, att_mask, text_words, start_positions, end_positions
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
