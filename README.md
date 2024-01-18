# PyTorch Implementation for "Better Explain Transformers by Illuminating Important Information"


### Datasets
We use preprocessed dataset parsed by Stanza with the data structure below:
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


### Details of our code.
`evaluate_explanability.py` is the code we use to run experiment of our Table 1, 2 and 3.

lambda_k in Equation (7) can be obtained by running `attn_head_stats.py`.

`self_parser.py` is the args for `evaluate_explanability.py`, where the value of `--synt-thres` corresponding to lambda_k + xi_synt (Equation (7)) in our manuscript and the value of `--pos-thres` corresponding to xi_pos in our manuscript.

The LRP implementation for each Transformer-based model is in `Transformer_Explanation`. 
We provide the implementation of BERT-base, RoBERTa-base and GPT-2.

Implementation of AOPC and LOdds are in `utils/metrices.py`.

