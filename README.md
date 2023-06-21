# PyTorch Implementation for "Better Explain Transformers by Illuminating Important Information"

## Introduction

### Datasets 
Datasets parsed by Stanza can be downloaded through the link below (SST-2, IMDB and SQUADv2, others will be available after the paper accepted; datasets are uploaded through the user named "Anonymous"):
https://drive.google.com/drive/folders/1R9HpF5_5CaR9ey5EHQozy4SIMP1Hy1nx

The above datasets have the following data structure (a sample in SST-2):
```
{
    "id": 0, 
    "sentence": "uneasy mishmash of styles and genres .", 
    "label": -1, 
    "parsed": [
        {"id": 1, "text": "uneasy", "lemma": "uneasy", "upos": "ADJ", "head": 2, "deprel": "amod"}, 
        {"id": 2, "text": "mishmash", "lemma": "mishmash", "upos": "NOUN", "head": 0, "deprel": "root"}, 
        {"id": 3, "text": "of", "lemma": "of", "upos": "ADP", "head": 4, "deprel": "case"}, 
        {"id": 4, "text": "styles", "lemma": "style", "upos": "NOUN", "head": 2, "deprel": "nmod"}, 
        {"id": 5, "text": "and", "lemma": "and", "upos": "CCONJ", "head": 6, "deprel": "cc"}, 
        {"id": 6, "text": "genres", "lemma": "genre", "upos": "NOUN", "head": 4, "deprel": "conj"}, 
        {"id": 7, "text": ".", "lemma": ".", "upos": "PUNCT", "head": 2, "deprel": "punct"}
    ]
}
```
Here, the key "sentence" may different in other dataset (e.g., "text" in Yelp and IMDB). The value of "parsed" is the parsed result of the input sentences from Stanza, which contain the word with its corresponding dependency relation (with the key "deprel", also called "syntactic relation" in our manuscript).


### Requirement
Install the pytorch (through the instruction in the official homepage) and other requirement packages by
```
pip install -r requirements.txt
```


### Dataset statistic for positional and syntactic information
Put the downloaded dataset into a folder and modify the `DATA_PATH` and all the save path at the final in `attn_head_stats.py` and run
```
python attn_head_stats.py
```


### Reproducing results in our paper
Run the comment below to get the evaluation result:
```
python test.py --num-process [number of process you want to use, default is 1] --devices [set your GPU ids devided by ,] --dataset [name of dataset] --expl-method [name of explanation method, options can be found by --help or in the file self_parser.py]
```
if you are running our method (MLRP), you need an additional args `--synt-thres`, which is related to the equation (7) in our paper and you can obtain the lambda_k from the statistic result in "Dataset statistic for positional and syntactic information".