from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from itertools import chain
from utils.metrices import truncate_words, replace_words, cal_kendaltau, cal_aopc, cal_logodds, cal_count, perfect_count
from self_parser import parser
from utils.preprocess import preprocess_sample, load_dataset_json, load_model_and_tokenizer, generate_head_mask
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import torch.multiprocessing as mp
import os
import json
import pickle as pkl
from scipy.special import softmax


os.environ['TOKENIZERS_PARALLELISM'] = 'true'
task_model = {
    "yelp": "fabriceyhc/bert-base-uncased-yelp_polarity",
    "imdb": "textattack/bert-base-uncased-imdb",
    "mnli": "textattack/bert-base-uncased-MNLI",
    "qqp": "modeltc/bert-base-uncased-qqp",
    "sst2": "textattack/bert-base-uncased-SST-2",
    "squadv1": "csarron/bert-base-uncased-squad-v1",
    "squadv2": "deepset/bert-base-uncased-squad2"
}

special_tokens = ["[CLS]", "[SEP]"]
special_idxs = [101, 102]
mask = "[PAD]"
mask_id = 0

upos_type = ['VERB', 'NOUN', 'ADV', 'ADJ', 'INTJ', 'PROPN']

error_cnt = 0
unseen_upos = set()
unseen_dtype = set()
error_data = {
    0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}, 10: {}, 11: {}
}



def generate_expl(input_ids, attention_mask, token_type_ids, target_class, head_mask, model, expl_method):
    expl_generator = Generator(model)
    token_type_ids = None

    # true class is positive - 1

    # generate an explanation for the input
    expl_func = getattr(expl_generator, expl_method, None)
    if expl_func == None:
        raise('Explanation method not exist.')
    elif expl_method == 'AttCAT':
        expl = expl_func(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index=target_class, head_mask=head_mask, token_type_ids=token_type_ids)
    else:
        expl = expl_func(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index=target_class, head_mask=head_mask, token_type_ids=token_type_ids)[0][0]

    return expl


def generate_qa_expl(input_ids, attention_mask, head_mask, model, expl_method, ans_index, is_start=True):
    expl_generator = Generator(model, is_qa=True, is_start=is_start)
    expl_func = getattr(expl_generator, expl_method, None)

    if expl_func == None:
        raise('Explanation method not exist.')
    elif expl_method == 'AttCAT':
        expl = expl_func(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index=None, head_mask=head_mask)
    else:
        expl = expl_func(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index=None)[0]

    return expl


def predict(model, text_ids, target, att_mask=None, seg_ids=None):
    seg_ids = None
    out = model(text_ids, attention_mask=att_mask, token_type_ids=seg_ids)
    prob = out[0]
    pred_class = torch.argmax(prob, axis=1).cpu().detach().numpy()
    pred_class_prob = softmax(prob.cpu().detach().numpy(), axis=1)
    
    return pred_class[0], pred_class_prob[:, target][0]


def calc_cls_metrics(dataset, data_name, model, tokenizer, head_mask=None, expl_method='generate_LRP', q=None, validation_label=None):

    original_probs = []
    degradation_probs = []
    kendaltaus = []

    for i, test_instance in enumerate(tqdm(dataset)):

        instance_degradation_probs = []
        instance_degradation_accs = []

        if data_name == 'qqp':
            input = test_instance
            target = validation_label[i]
        elif data_name == 'mnli':
            input = test_instance
            target = test_instance['label']
        elif data_name in ['yelp', 'imdb']:
            input = test_instance['text']
            target = test_instance['label']
        else:
            input = test_instance['sentence']
            target = test_instance['label']
        
        input_ids, att_mask, text_words, token_type_ids = preprocess_sample(input, tokenizer, model.device, data_name)
        
        # get truc words number
        total_len = len(text_words)
        if total_len < 20:
            continue

        granularity = np.linspace(0, 1, 10)
        trunc_words_num = [int(g) for g in np.round(granularity * total_len)]
        trunc_words_num = list(dict.fromkeys(trunc_words_num))
        
        _, original_prob = predict(model, input_ids, target, seg_ids=token_type_ids)

        # if data_name == 'qqp' and (target == 1 or original_prob < 0.5):
        #     continue

        # if data_name == 'mnli' and original_prob < 0.34:  # make sure the model make an accurate prediction.
        #     continue

        expl = generate_expl(input_ids, att_mask, token_type_ids, target, head_mask, model, expl_method)
        expl_F = generate_expl(input_ids, att_mask, token_type_ids, 1 - target, head_mask, model, expl_method)
        sorted_idx = np.argsort(-expl)

        for num in trunc_words_num[1:]:
            replaced_text_ids = replace_words(sorted_idx, text_words, input_ids, num, special_tokens=special_tokens)

            rep_class, rep_prob = predict(model, replaced_text_ids, target, seg_ids=token_type_ids)

            instance_degradation_probs.append(rep_prob)
            instance_degradation_accs.append(rep_class == target)

        original_probs.append(original_prob)
        degradation_probs.append(instance_degradation_probs)
        kendaltau = cal_kendaltau(expl, expl_F)
        kendaltaus.append(kendaltau)

    aopc = cal_aopc(original_probs, degradation_probs)
    logodds = cal_logodds(original_probs, degradation_probs)

    if q is not None:
        q.put((aopc, [logodds], kendaltaus))
    else:
        return aopc, [logodds], kendaltaus


def calc_qa_metrics(dataset, data_name, model, tokenizer, head_mask=None, expl_method='generate_LRP', q=None):
    scores = []
    perfect_scores = []
    
    for i, test_instance in enumerate(tqdm(dataset)):
        torch.cuda.empty_cache()

        input_ids, att_mask, _, start_positions, end_positions = preprocess_sample(test_instance, tokenizer, model.device, data_name)
        expln_start = generate_qa_expl(input_ids, att_mask, head_mask, model, expl_method, start_positions, is_start=True)
        expln_end = generate_qa_expl(input_ids, att_mask, head_mask, model, expl_method, end_positions, is_start=False)
        expln = expln_start + expln_end

        count = cal_count(start_positions[0], end_positions[0], expln)
        p_count = perfect_count(start_positions[0], end_positions[0])

        scores.append(count)
        perfect_scores.append(p_count)
    
    if q is not None:
        q.put((np.mean(scores), np.mean(perfect_scores)))
    else:
        return np.mean(scores), np.mean(perfect_scores)


if __name__ == '__main__':
    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')
    models = []

    devices = [f'cuda:{e}' for e in args.devices.split(',')]
    dataset = args.dataset
    num_p = args.num_process
    expl_method = args.expl_method
    synt_acc_threshold = [float(e) for e in args.synt_thres.split(',')]
    pos_threshold = args.pos_thres
    upos_threshold = args.upos_thres
    parsed_path = args.parsed_path
    data_path = args.data_path
    mask_type = args.mask_type
    save_path = args.save_path

    for d in devices:
        if dataset in ['squadv1', 'squadv2']:
            model, tokenizer = load_model_and_tokenizer(task_model[dataset], device=d, is_qa=True)
        else:
            model, tokenizer = load_model_and_tokenizer(task_model[dataset], device=d)
        models.append(model)

    tokenized_qqp = None
    validation_label = []
    if dataset == 'qqp':
        qqp = load_dataset('glue', dataset)
        for data in qqp['validation']:
            validation_label.append(data['label'])

    if dataset == 'mnli':
        valid = load_dataset_json(type='validation_matched', dataset=dataset, data_path=data_path)
    elif dataset in ['yelp', 'imdb']:
        valid = load_dataset_json(type='test', dataset=dataset, data_path=data_path)
    else:
        valid = load_dataset_json(type='validation', dataset=dataset, data_path=data_path)

    print('loading deprel res from cache...')

    with open(f'{parsed_path}{dataset}_total_deprel_stats_res.pkl', 'rb') as f:
        dtype_list_relpos = pkl.load(f)

    with open(f'{parsed_path}{dataset}_total_deprel_hit_res.pkl', 'rb') as f:
        deprel_hit = pkl.load(f)

    with open(f'{parsed_path}{dataset}_total_deprel_miss_res.pkl', 'rb') as f:
        deprel_miss = pkl.load(f)

    with open(f'{parsed_path}{dataset}_rel_ppos_res.pkl', 'rb') as f:
        rel_ppos_res = pkl.load(f)

    with open(f'{parsed_path}{dataset}_rel_npos_res.pkl', 'rb') as f:
        rel_npos_res = pkl.load(f)

    with open(f'{parsed_path}{dataset}_total_upos_res.pkl', 'rb') as f:
        upos_res = pkl.load(f)

    # syntactic
    deprel_total = deprel_hit + deprel_miss
    deprel_acc = deprel_hit / deprel_total
    deprel_error = deprel_miss / deprel_total

    # positional
    total_pos_stats = (rel_ppos_res + rel_npos_res).sum(axis=2)[0][0]
    ppos_freq = rel_ppos_res / total_pos_stats
    npos_freq = rel_npos_res / total_pos_stats

    # syntactic mask
    synt_mask_obj = np.argwhere(deprel_acc > synt_acc_threshold[0])[np.where(np.argwhere(deprel_acc > synt_acc_threshold[0])[:, 2] == 0)]
    synt_mask_nsubj = np.argwhere(deprel_acc > synt_acc_threshold[1])[np.where(np.argwhere(deprel_acc > synt_acc_threshold[1])[:, 2] == 1)]
    synt_mask_amod = np.argwhere(deprel_acc > synt_acc_threshold[2])[np.where(np.argwhere(deprel_acc > synt_acc_threshold[2])[:, 2] == 2)]
    synt_mask_advmod = np.argwhere(deprel_acc > synt_acc_threshold[3])[np.where(np.argwhere(deprel_acc > synt_acc_threshold[3])[:, 2] == 3)]

    # pos mask
    pos_mask = np.vstack([np.argwhere(ppos_freq > pos_threshold), np.argwhere(npos_freq > pos_threshold)])

    # upos mask
    for i, u in enumerate(upos_res.sum(axis=2)):
        for j, u2 in enumerate(u):
            upos_res[i, j, :] = upos_res[i, j, :] / u2
    upos_mask_total = []
    for i in range(len(upos_type)):
        upos_mask_total.append(np.argwhere(upos_res > upos_threshold)[np.where(np.argwhere(upos_res > upos_threshold)[:, 2] == i)])

    if mask_type == 'synt':
        mask = np.vstack([synt_mask_obj, synt_mask_nsubj, synt_mask_amod, synt_mask_advmod])
    elif mask_type == 'synt_pos':
        mask = np.vstack([synt_mask_obj, synt_mask_nsubj, synt_mask_amod, synt_mask_advmod, pos_mask])
    elif mask_type == 'synt_upos':
        mask = np.vstack(upos_mask_total + [synt_mask_obj, synt_mask_nsubj, synt_mask_amod, synt_mask_advmod])
    elif mask_type == 'upos':
        mask = np.vstack(upos_mask_total)
    elif mask_type == 'upos_pos':
        mask = np.vstack(upos_mask_total + [pos_mask])
    elif mask_type == 'pos':
        mask = np.vstack([pos_mask])
    elif mask_type == 'all':
        mask = np.vstack([synt_mask_obj, synt_mask_nsubj, synt_mask_amod, synt_mask_advmod, pos_mask] + upos_mask_total)
    else:
        mask = None

    if mask_type == 'random':
        head_mask = torch.tensor(generate_head_mask(type='random', mask_indices=mask, num_layers=12, num_heads=12))
    elif mask_type == 'orig':
        head_mask = None
    else:
        head_mask = torch.tensor(generate_head_mask(type='indices', mask_indices=mask, num_layers=12, num_heads=12))

    results = []
    q = mp.Queue()
    processes = []
    if num_p > 1:
        chunk_size = len(valid) // num_p
        chunks = [valid[i:i+chunk_size] for i in range(0, len(valid), chunk_size)]
        for i, chunk in enumerate(chunks):
            # distribute chunks to different model in different process
            model = models[i % len(models)]
            if head_mask != None:
                head_mask = head_mask.to(model.device)
            if dataset in ['squadv1', 'squadv2']:
                p = mp.Process(target=calc_qa_metrics, args=(chunk, dataset, model, tokenizer, head_mask, expl_method, q))
            else:
                p = mp.Process(target=calc_cls_metrics, args=(chunk, dataset, model, tokenizer, head_mask, expl_method, q, validation_label))
            p.start()
        while len(results) < len(chunks):
            if not q.empty():
                results.append(q.get())
        print('all done')
        
    else:
        if head_mask != None:
            head_mask = head_mask.to(models[0].device)
        if dataset in ['squadv1', 'squadv2']:
            results = [calc_qa_metrics(valid, dataset, models[0], tokenizer, head_mask, expl_method)]
        else:
            results = [calc_cls_metrics(valid, dataset, models[0], tokenizer, head_mask, expl_method, validation_label=validation_label)]

    if dataset in ['squadv1', 'squadv2']:
        score = []
        p_score = []
        for result in results:
            score.append(result[0])
            p_score.append(result[1])
        print(np.mean(score), np.mean(p_score))
        save_dict = {
            'score': np.mean(score),
            'perfect_score': np.mean(p_score),
        }
        with open(f'{save_path}/{dataset}_{expl_method}_{mask_type}.json', 'w') as f:
            json.dump(save_dict, f, indent=4)

    else:
        aopc = []
        logodds = []
        kendaltaus = []
        for result in results:
            aopc.append(result[0])
            logodds += result[1]
            kendaltaus += result[2]

        aopc = np.mean(aopc)
        logodds = np.mean(logodds)
        k = np.mean(kendaltaus)

        print(aopc, logodds, k)
        save_dict = {
            'aopc': str(aopc),
            'logodds': str(logodds),
            'k': str(k)
        }
        with open(f'{save_path}/{dataset}_{expl_method}_{mask_type}.json', 'w') as f:
            json.dump(save_dict, f, indent=4)

