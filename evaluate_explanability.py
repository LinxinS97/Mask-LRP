from Transformer_Explanation.ExplanationGenerator import Generator
from utils.metrices import replace_words, cal_aopc, cal_logodds, cal_count, perfect_count
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
    "squadv1": {
        "bert": "csarron/bert-base-uncased-squad-v1",
        "gpt2": 'anas-awadalla/gpt2-span-head-finetuned-squad',
        "roberta": "thatdramebaazguy/roberta-base-squad"
    },
    "squadv2": {
        "bert": "ericRosello/bert-base-uncased-finetuned-squad-frozen-v2",
        "gpt2": "anas-awadalla/gpt2-span-head-finetuned-squad",
        "roberta": "21iridescent/roberta-base-finetuned-squad2-lwt"
    }
}


def generate_expl(input_ids, attention_mask, token_type_ids, target_class, head_mask, model, expl_method):
    expl_generator = Generator(model)
    token_type_ids = None

    expl_func = getattr(expl_generator, expl_method, None)
    if expl_func == None:
        raise('Explanation method not exist.')
    elif expl_method == 'AttCAT':
        expl = expl_func(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index=target_class, head_mask=head_mask, token_type_ids=token_type_ids)
    else:
        expl = expl_func(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index=target_class, head_mask=head_mask, token_type_ids=token_type_ids)[0][0]

    return expl


def generate_qa_expl(input_ids, attention_mask, head_mask, model, expl_method, ans_index, is_start=True, model_name='bert'):
    expl_generator = Generator(model, is_qa=True, is_start=is_start, model_name=model_name)
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

        # For duplicated quesion matching, model will turn the prediction to 0 if we prune all influencial tokens
        # But if the ground truth is initially 1, model's confidence will not change.
        # So we only test the matched data with ground truth label equals to 1.
        if data_name == 'qqp' and target == 0:
            continue
        
        # get truc words number
        total_len = len(text_words)
        if total_len < 20:
            continue

        granularity = np.linspace(0, 1, 10)
        trunc_words_num = [int(g) for g in np.round(granularity * total_len)]
        trunc_words_num = list(dict.fromkeys(trunc_words_num))
        
        _, original_prob = predict(model, input_ids, target, seg_ids=token_type_ids)
        
        
        # make sure the model make an accurate prediction.
        if data_name == 'qqp' and original_prob < 0.3:
            continue
        if data_name == 'mnli' and original_prob < 0.34:
            continue

        expl = generate_expl(input_ids, att_mask, token_type_ids, target, head_mask, model, expl_method)
        sorted_idx = np.argsort(-expl)

        for num in trunc_words_num[1:]:
            replaced_text_ids = replace_words(sorted_idx, text_words, input_ids, num, data_name=data_name)

            rep_class, rep_prob = predict(model, replaced_text_ids, target, seg_ids=token_type_ids)

            instance_degradation_probs.append(rep_prob)
            instance_degradation_accs.append(rep_class == target)

        original_probs.append(original_prob)
        degradation_probs.append(instance_degradation_probs)

    aopc = cal_aopc(original_probs, degradation_probs)
    logodds = cal_logodds(original_probs, degradation_probs)

    if q is not None:
        q.put((aopc, logodds))
    else:
        return aopc, logodds


def calc_qa_metrics(dataset, data_name, model, tokenizer, head_mask=None, expl_method='generate_LRP', q=None, model_name='bert'):
    scores = []
    perfect_scores = []
    
    for i, test_instance in enumerate(tqdm(dataset)):
        torch.cuda.empty_cache()

        try:
            input_ids, att_mask, _, start_positions, end_positions = preprocess_sample(test_instance, tokenizer, model.device, data_name)
        except Exception:
            continue
        expln_start = generate_qa_expl(input_ids, att_mask, head_mask, model, expl_method, start_positions, is_start=True, model_name=model_name)
        expln_end = generate_qa_expl(input_ids, att_mask, head_mask, model, expl_method, end_positions, is_start=False, model_name=model_name)
        expln = expln_start + expln_end

        count = cal_count(start_positions[0], end_positions[0], expln)
        p_count = perfect_count(start_positions[0], end_positions[0])

        scores.append(count)
        perfect_scores.append(p_count)
    
    if q is not None:
        q.put((scores, perfect_scores))
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
    model_name = args.model_name
    corruption_rate = args.corruption_rate
    repeat = args.repeat


    if dataset == 'qqp':
        special_tokens = []
        special_idxs = []

    for d in devices:
        if dataset in ['squadv1', 'squadv2']:
            model, tokenizer = load_model_and_tokenizer(task_model[dataset][model_name], device=d, is_qa=True)
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

    # Repeat n times if we use randomly generated head mask.
    # all results will be saved in a single file.
    all_res = []
    for i in range(repeat):
        if mask_type == 'orig':
            head_mask = None
        else:
            head_mask = generate_head_mask(parsed_path, dataset, model_name, 
                                        mask_type, 
                                        synt_acc_threshold, pos_threshold,
                                        corruption_rate=corruption_rate,
                                        num_heads=model.config.num_attention_heads, num_layers=model.config.num_hidden_layers)

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
                    p = mp.Process(target=calc_qa_metrics, args=(chunk, dataset, model, tokenizer, head_mask, expl_method, q, model_name))
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
                results = [calc_qa_metrics(valid, dataset, models[0], tokenizer, head_mask, expl_method, model_name=model_name)]
            else:
                results = [calc_cls_metrics(valid, dataset, models[0], tokenizer, head_mask, expl_method, validation_label=validation_label)]

        if dataset in ['squadv1', 'squadv2']:
            score = []
            p_score = []
            for result in results:
                score += result[0]
                p_score += result[1]
            print(np.mean(score), np.mean(p_score), np.mean(score) / np.mean(p_score))
            save_dict = {
                'score': np.mean(score),
                'perfect_score': np.mean(p_score),
                'precision@20': np.mean(score) / np.mean(p_score)
            }
            with open(f'{save_path}/{dataset}_{model_name}_{expl_method}_{mask_type}.json', 'w') as f:
                json.dump(save_dict, f, indent=4)

        else:
            aopc = []
            logodds = []
            kendaltaus = []
            for result in results:
                aopc.append(result[0])
                logodds.append(result[1])

            aopc = np.array(aopc).mean(axis=0).tolist()
            logodds = np.array(logodds).mean(axis=0).tolist()

            print(np.mean(aopc), np.mean(logodds))

            save_dict = {
                'aopc': aopc,
                'logodds': logodds,
                'aopc_avg': np.mean(aopc),
                'logodds_avg': np.mean(logodds)
            }
            all_res.append(save_dict)

    if mask_type == 'synt_pos_corruption':
        mask_type = mask_type + '_' + str(corruption_rate)
    with open(f'{save_path}/{dataset}_{expl_method}_{mask_type}.json', 'w') as f:
        json.dump(all_res, f, indent=4)

