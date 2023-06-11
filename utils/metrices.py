import numpy as np
import math
import scipy.stats as stats


def cal_aopc(original_probs, degradation_probs):
    original_probs = np.array(original_probs)
    degradation_probs = np.array(degradation_probs)
    
    diffs = []
    for i in range(len(original_probs)):
        diffs_k = []
        for j in range(9):
            diff = original_probs[i] - degradation_probs[i][j]
            diffs_k.append(diff)
        diffs.append(diffs_k)

    result = np.mean(diffs, axis=0)
    # aopc = np.mean(result)
    
    return result


def cal_logodds(original_probs, degradation_probs):
    original_probs = np.array(original_probs)
    degradation_probs = np.array(degradation_probs)
    
    ratios = []
    for i in range(len(original_probs)):
        ratios_k = []
        for j in range(9):
            ratio = math.log(degradation_probs[i][j] / original_probs[i])
            ratios_k.append(ratio)
        ratios.append(ratios_k)

    result = np.mean(ratios, axis=0)
    # logodds = np.mean(result)
    
    return result


def truncate_words(sorted_idx, text_words, text_ids, replaced_num, seg_ids=None, special_tokens=None):
    to_be_replaced_idx = []
    i = 0
    while len(to_be_replaced_idx) < replaced_num and i != len(text_words) - 1:
        current_idx = sorted_idx[i]
        if text_words[current_idx] not in special_tokens:
            to_be_replaced_idx.append(current_idx)
        i += 1
    remaining_idx = sorted(list(set(sorted_idx) - set(to_be_replaced_idx)))
    truncated_text_ids = text_ids[0, np.array(remaining_idx)]
    if seg_ids is not None:
        seg_ids = seg_ids[0, np.array(remaining_idx)]
    truncated_text_words = np.array(text_words)[remaining_idx]
    return truncated_text_ids.unsqueeze(0), truncated_text_words, seg_ids


def replace_words(sorted_idx, text_words, text_ids, replaced_num, data_name=None):
    
    special_tokens = ["[CLS]", "[SEP]"]
    mask_id = 0
    to_be_replaced_idx = []

    i= 0
    while len(to_be_replaced_idx) < replaced_num and i != len(sorted_idx) - 1:
        current_idx = sorted_idx[i]
        if text_words[current_idx] not in special_tokens:
            to_be_replaced_idx.append(current_idx)
        i += 1
    replaced_text_ids = text_ids.clone()
    replaced_text_ids[0, to_be_replaced_idx] = mask_id

    return replaced_text_ids

    sorted_idx1 = np.argsort(-attribution1)
    sorted_idx2 = np.argsort(-attribution2)

    tau, p_value = stats.kendalltau(sorted_idx1, sorted_idx2)
    
    return tau


def cal_count(start_positions, end_positions, attribution):
    answers = []
    sorted_idx = np.argsort(-attribution)
    sorted_idx = set(sorted_idx[:20])
    
    for i in range(start_positions, end_positions + 1):
        answers.append(i)
        
    count = 0
    
    for a in answers:
        if a in sorted_idx:
            count +=1 
        
    return count


def perfect_count(start_positions, end_positions):
    
    answers = []
    for i in range(start_positions, end_positions+1):
        answers.append(i)
        
    count = len(answers)
    
    return count
