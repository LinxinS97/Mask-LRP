from itertools import chain
from typing import List
import torch
import string
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import torch.multiprocessing as mp
import os
from os.path import exists
import pickle as pkl
import re
from utils.preprocess import preprocess_sample, load_dataset_json, load_model_and_tokenizer


os.environ['TOKENIZERS_PARALLELISM'] = 'true'
DATASET = 'squadv2'  # yelp qqp imdb squadv1 squadv2 mnli
DEVICE_NUM = 1
DATA_PATH = f'/home/data'
DEVICE = torch.device(f'cuda:{DEVICE_NUM}')  # 2 and 3 is available

task_to_keys = {
    "imdb": ("text", None),
    "yelp": ("text", None),
    "mnli": ("premise", "hypothesis"),
    "qqp": ("question1", "question2"),
    "sst2": ("sentence", None),
    "squadv1": ("context", "question"),
    "squadv2": ("context", "question")
}

deprel_type = ['nsubj', 'obj', 'amod', 'advmod']
upos_type = ['VERB', 'NOUN', 'ADV', 'ADJ', 'INTJ', 'PROPN']
deprel_map = {
    'nsubj': 0,
    'obj': 1,
    'amod': 2,
    'advmod': 3
}

MAX_SENTENCE_LEN = 600  # 600 for yelp
error_cnt = 0
unseen_upos = set()
unseen_dtype = set()
error_data = {
    0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}, 10: {}, 11: {}
}

punc_dict = str.maketrans({ p: "[MASK]" for p in string.punctuation })



def get_assembled_token(tokens) -> List[tuple]:
    res = []
    subword_mask = []
    subword_mask_tmp = []
    for idx, t in enumerate(tokens):
            idx = idx + 1
            if '##' in t:
                subword_mask_tmp.append(idx)
                if idx == len(tokens):
                    subword_mask.append([x - 1 for x in subword_mask_tmp])
            else:
                if len(subword_mask_tmp) > 1:
                    subword_mask.append([x - 1 for x in subword_mask_tmp])
                subword_mask_tmp = []
                subword_mask_tmp.append(idx)
                
    for pos, token in enumerate(tokens):
        pos = (pos, )
        if '##' in token:
            continue
        for m in subword_mask:
            if pos[0] == m[0]:
                token = tokens[m[0]] + ''.join([tokens[p][2:] for p in m[1:]])  # e.g. 'secret' + '##ion', remove '##' and add them on
                pos = m
                break
        
        token = token.replace('[CLS]', '')
        token = token.replace('[SEP]', '')
        token = token.replace('[MASK]', '')
        token = token.translate(str.maketrans('', '', string.punctuation))
        res.append((pos, token))
    return res


def get_gpt_assembled_token(tokens) -> List[tuple]:
    res = []
    subword_mask = []
    subword_mask_tmp = []
    for idx, t in enumerate(tokens):
        idx = idx + 1
        if 'Ġ' not in t and not t[0].isupper():
            subword_mask_tmp.append(idx)
            if idx == len(tokens):
                subword_mask.append([x - 1 for x in subword_mask_tmp])
        else:
            if len(subword_mask_tmp) > 0:
                subword_mask.append([x - 1 for x in subword_mask_tmp])
            subword_mask_tmp = []
            subword_mask_tmp.append(idx)
                
    for pos, token in enumerate(tokens):
        pos = (pos, )
        if 'Ġ' not in token and not token[0].isupper():
            continue
        for m in subword_mask:
            if pos[0] == m[0]:
                if (token[0].isupper() and token[0] != 'Ġ') or token == '[MASK]':
                    token = tokens[m[0]] + ''.join([tokens[p] for p in m[1:]])
                else:
                    token = tokens[m[0]][1:] + ''.join([tokens[p] for p in m[1:]])
                pos = m
                break
        if len(list(filter(None, token.split('[MASK]')))) > 1:
            tmp = list(filter(None, token.split('[MASK]')))
            pos_id = 0
            for idx, t in enumerate(tmp):
                res.append(((pos[pos_id], ), t))
                pos_id += 2
        else:
            token = token.replace('[MASK]', '')
            token = token.translate(str.maketrans('', '', string.punctuation))
            res.append((pos, token))
    return res


def find_assembled_position(assembled_toknes, key_word):
    position, tokens = zip(*assembled_toknes)
    try:
        return position[tokens.index(key_word)]
    except Exception:
        return -1
    

def find_assembled_token(assembled_toknes, key_pos):
    
    position, tokens = zip(*assembled_toknes)
    for i, poss in enumerate(position):
        if key_pos in poss:
            return tokens[i]
    return ''


def get_lexical_type(token, deprel):
    w_type = None
    for d in deprel:
        if d['text'].lower() == token.lower():
            w_type = d['upos']
            break
    return w_type


def element_count(indices, target):
    unique_indices, counts = np.unique(indices, return_counts=True)
    target[unique_indices] += counts

    return target


def multiproc_deprel(dataset, tokenizer, model, input_type='sentence', q=None, model_name='bert'):
    print(f'Process ID: {os.getpid()}, start')

    deprel_hit = np.zeros((model.config.num_hidden_layers, model.config.num_attention_heads, len(deprel_map.keys())))
    deprel_miss = np.zeros((model.config.num_hidden_layers, model.config.num_attention_heads, len(deprel_map.keys())))
    upos_res = np.zeros((model.config.num_hidden_layers, model.config.num_attention_heads, len(upos_type)))
    rel_ppos_res = np.zeros((model.config.num_hidden_layers, model.config.num_attention_heads, MAX_SENTENCE_LEN))  # positive relative position
    rel_npos_res = np.zeros((model.config.num_hidden_layers, model.config.num_attention_heads, MAX_SENTENCE_LEN))  # negative relative position
    for data in tqdm(dataset):
        # process data
        if DATASET == 'qqp':
            deprels = [*data['parsed1'], *data['parsed2']]
            input = data
        elif DATASET == 'mnli':
            deprels = [data['parsed_premise'], data['parsed_hypothesis']]
            input = data
        elif DATASET == 'squadv1':
            deprels = [*data['parsed_question'], *data['parsed_context']]
            input = data
        elif DATASET == 'squadv2':
            deprels = [*data['question_parsed'], *data['context_parsed']]
            input = data
        elif DATASET in ['yelp', 'imdb']:
            deprels = data['parsed']
            input = data[input_type]
        else:
            deprels = [data['parsed']]
            input = data[input_type]

        token_type_ids = None
        if DATASET in ['squadv1', 'squadv2']:
            try:
                input_ids, attention_mask, tokens, start_positions, end_positions = preprocess_sample(input, tokenizer, device=DEVICE, dataset=DATASET)
            except Exception:
                continue
        else:
            input_ids, attention_mask, tokens, token_type_ids = preprocess_sample(input, tokenizer, device=DEVICE, dataset=DATASET)
        
        punc_idx = np.where(np.array([t in string.punctuation for t in tokens]) == True)[0]
        for i in punc_idx:
            tokens[i] = '[MASK]'
        
        if 'gpt' not in model_name:
            assembled_tokens = get_assembled_token(tokens)  # aggregate subwords and their corresponding positions
        else:
            assembled_tokens = get_gpt_assembled_token(tokens)  # aggregate subwords and their corresponding positions

        # get attn output
        attns = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, head_mask=None, output_attentions=True).attentions

        for layer_id, attn in enumerate(attns):
            attn = attn.squeeze().detach().cpu().numpy()
            for head_id, head in enumerate(attn):
                
                max_attn_pos = np.argmax(head, axis=1)
                rel_pos = max_attn_pos - np.arange(max_attn_pos.shape[0])
                rel_npos_res[layer_id, head_id, :] = element_count(rel_pos[rel_pos < 0] * -1, rel_npos_res[layer_id, head_id])
                rel_ppos_res[layer_id, head_id, :] = element_count(rel_pos[rel_pos >= 0], rel_ppos_res[layer_id, head_id])
                
                max_attn_token = [find_assembled_token(assembled_tokens, pos) for pos in max_attn_pos]
                
                idx_token = 0
                pos = 0
                for deprel_idx in range(len(deprels)):
                    idx_word = 0

                    while idx_word < len(deprels[deprel_idx]):
                        word = deprels[deprel_idx][idx_word]
                        word_t = word['text'].translate(punc_dict)
                        if '[MASK]' in word_t or (word_t[0].isdigit() and word_t[-1].isdigit()) or len(word_t) == 1:
                            idx_word += 1
                            continue

                        if pos == len(assembled_tokens) - 1:
                            break
                        p, token = assembled_tokens[pos]
                        if token == '' or token.isdigit() or len(token) == 1:
                            pos += 1
                            if pos == len(assembled_tokens) - 1:
                                break
                            continue
                        idx_token = p[0]
                        pos += 1
                        if pos == len(assembled_tokens) - 1:
                            break

                        d_type, head = word['deprel'], deprels[deprel_idx][word['head'] - 1]['text']
                        if d_type in deprel_type:
                            if head.lower() == max_attn_token[idx_token].replace('[MASK]', ''):
                                deprel_hit[layer_id, head_id, deprel_map[d_type]] += 1
                            else:
                                deprel_miss[layer_id, head_id, deprel_map[d_type]] += 1

                        idx_word += 1
                        if idx_token < len(tokens) - 1:
                            idx_token += 1

    if q != None:
        q.put((deprel_hit, deprel_miss, rel_ppos_res, rel_npos_res, upos_res))
        print(f'Process ID: {os.getpid()}, done')
        return
    else:
        return deprel_hit, deprel_miss, rel_ppos_res, rel_npos_res, upos_res


def deprel_stats(dataset, stats_res):
    for data in dataset:
        if DATASET == 'qqp':
            deprels = [data['parsed1'], data['parsed2']]
        elif DATASET == 'mnli':
            deprels = [[data['parsed_premise'], data['parsed_hypothesis']]]
        elif DATASET == 'squadv1':
            deprels = [data['parsed_context'], data['parsed_context']]
        elif DATASET == 'squadv2':
            deprels = [data['context_parsed'], data['context_parsed']]
        elif DATASET in ['yelp', 'imdb']:
            deprels = [data['parsed']]
        else:
            deprels = [[data['parsed']]]
        for deprel in deprels:
            for d_type in list(chain(*deprel)):
                if d_type['deprel'] in deprel_type:
                    rel_pos = d_type['id'] - d_type['head']
                    stats_res[d_type['deprel']].append(rel_pos)



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    num_p = 10
    input_type = 'text'
    stats_res = {}
    for d_type in deprel_map.keys():
        stats_res.setdefault(d_type, [])

    train = load_dataset_json(type='train', dataset=DATASET, data_path=DATA_PATH)
    valid = load_dataset_json(type='validation', dataset=DATASET, data_path=DATA_PATH)
    # test = load_dataset_json(type='test', dataset=DATASET, data_path=DATA_PATH)

    dataset = train + valid
    deprel_stats(dataset, stats_res)

    # SST2: textattack/bert-base-uncased-SST-2
    # IMDB: textattack/bert-base-uncased-imdb
    # YELP: fabriceyhc/bert-base-uncased-yelp_polarity
    # QQP: modeltc/bert-base-uncased-qqp
    # MNLI: textattack/bert-base-uncased-MNLI

    # SQUADv1 (bert base): csarron/bert-base-uncased-squad-v1
    # SQUADv1 (GPT2): anas-awadalla/gpt2-span-head-finetuned-squad
    # SQUADv1 (roberta): thatdramebaazguy/roberta-base-squad

    # SQUADv2 (bert base): ericRosello/bert-base-uncased-finetuned-squad-frozen-v2
    # SQUADv2 (GPT2): anas-awadalla/gpt2-span-head-finetuned-squad
    # SQUADv2 (roberta): 21iridescent/RoBERTa-base-finetuned-squad2-lwt
    model_name = 'ericRosello/bert-base-uncased-finetuned-squad-frozen-v2'
    model, tokenizer = load_model_and_tokenizer(model_name, device=DEVICE, is_qa=True)

    deprel_hit = np.zeros((model.config.num_hidden_layers, model.config.num_attention_heads, len(deprel_map.keys())))
    deprel_miss = np.zeros((model.config.num_hidden_layers, model.config.num_attention_heads, len(deprel_map.keys())))
    upos_res = np.zeros((model.config.num_hidden_layers, model.config.num_attention_heads, len(upos_type)))
    rel_ppos_res = np.zeros((model.config.num_hidden_layers, model.config.num_attention_heads, MAX_SENTENCE_LEN))  # positive relative position
    rel_npos_res = np.zeros((model.config.num_hidden_layers, model.config.num_attention_heads, MAX_SENTENCE_LEN))  # negative relative position

    results = []
    q = mp.Queue()
    processes = []
    if num_p > 1:
        chunk_size = len(dataset) // num_p
        chunks = [dataset[i:i+chunk_size] for i in range(0, len(dataset), chunk_size)]
        for chunk in chunks:
            p = mp.Process(target=multiproc_deprel, args=(chunk, tokenizer, model, input_type, q, model_name))
            p.start()
        while len(results) < len(chunks):
            if not q.empty():
                results.append(q.get())
        print('all done')
    else:
        results = [multiproc_deprel(dataset, tokenizer, model, input_type)]
    
    for result in results:
        deprel_hit += result[0]
        deprel_miss += result[1]
        rel_ppos_res += result[2]
        rel_npos_res += result[3]

    _model = ''
    if DATASET in ['squadv1', 'squadv2']:
        _model = f'_{model_name.split("/")[1].split("-")[0]}'
    
    with open(f'/home/xai/Mask_LRP/parse_res/{DATASET}{_model}_total_deprel_stats_res.pkl', 'wb') as f:
        pkl.dump(stats_res, f)

    with open(f'/home/xai/Mask_LRP/parse_res/{DATASET}{_model}_total_deprel_hit_res.pkl', 'wb') as f:
        pkl.dump(deprel_hit, f)

    with open(f'/home/xai/Mask_LRP/parse_res/{DATASET}{_model}_total_deprel_miss_res.pkl', 'wb') as f:
        pkl.dump(deprel_miss, f)

    with open(f'/home/xai/Mask_LRP/parse_res/{DATASET}{_model}_rel_ppos_res.pkl', 'wb') as f:
        pkl.dump(rel_ppos_res, f)

    with open(f'/home/xai/Mask_LRP/parse_res/{DATASET}{_model}_rel_npos_res.pkl', 'wb') as f:
        pkl.dump(rel_npos_res, f)

