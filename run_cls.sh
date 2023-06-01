DEVICE="0,2"

# # SST2
# nohup python test.py --num-process 10 --mask-type "orig" --devices $DEVICE --dataset "sst2" --expl-method "generate_LRP" &
# wait
# nohup python test.py --num-process 10 --mask-type "synt_pos" --devices $DEVICE --dataset "sst2" --expl-method "generate_LRP" --synt-thres "0.46,0.43,0.76,0.72" &
# wait
# nohup python test.py --num-process 10 --mask-type "upos_pos" --devices $DEVICE --dataset "sst2" --expl-method "generate_LRP" &
# wait
# nohup python test.py --num-process 10 --devices $DEVICE --dataset "sst2" --expl-method "AttCAT" &
# wait

# # MNLI
# nohup python test.py --num-process 10 --mask-type "orig" --devices $DEVICE --dataset "mnli" --expl-method "generate_LRP" &
# wait
# nohup python test.py --num-process 10 --mask-type "synt_pos" --devices $DEVICE --dataset "mnli" --expl-method "generate_LRP" --synt-thres "0.49,0.44,0.84,0.56" &
# wait
# nohup python test.py --num-process 10 --mask-type "upos_pos" --devices $DEVICE --dataset "mnli" --expl-method "generate_LRP" &
# wait
# nohup python test.py --num-process 10 --devices $DEVICE --dataset "mnli" --expl-method "AttCAT" &
# wait

# QQP
nohup python test.py --num-process 10 --mask-type "orig" --devices $DEVICE --dataset "qqp" --expl-method "generate_LRP" &
wait
nohup python test.py --num-process 10 --mask-type "synt_pos" --devices $DEVICE --dataset "qqp" --expl-method "generate_LRP" --synt-thres "0.55,0.39,0.82,0.45" &
wait
nohup python test.py --num-process 10 --mask-type "upos_pos" --devices $DEVICE --dataset "qqp" --expl-method "generate_LRP" &
wait
nohup python test.py --num-process 10 --devices $DEVICE --dataset "qqp" --expl-method "AttCAT" &
wait

# # YELP
# nohup python test.py --num-process 10 --mask-type "orig" --devices $DEVICE --dataset "yelp" --expl-method "generate_LRP" &
# wait
# nohup python test.py --num-process 10 --mask-type "synt_pos" --devices $DEVICE --dataset "yelp" --expl-method "generate_LRP" --synt-thres "0.46,0.45,0.82,0.46" &
# wait
# nohup python test.py --num-process 10 --mask-type "upos_pos" --devices $DEVICE --dataset "yelp" --expl-method "generate_LRP" &
# wait
# nohup python test.py --num-process 10 --devices $DEVICE --dataset "yelp" --expl-method "AttCAT" &
# wait

# # IMDB
# nohup python test.py --num-process 10 --mask-type "orig" --devices $DEVICE --dataset "imdb" --expl-method "generate_LRP" &
# wait
# nohup python test.py --num-process 10 --mask-type "synt_pos" --devices $DEVICE --dataset "imdb" --expl-method "generate_LRP" --synt-thres "0.44,0.44,0.80,0.58" &
# wait
# nohup python test.py --num-process 10 --mask-type "upos_pos" --devices $DEVICE --dataset "imdb" --expl-method "generate_LRP" &
# wait
# nohup python test.py --num-process 10 --devices $DEVICE --dataset "imdb" --expl-method "AttCAT" &
# wait

# # squadv1
# nohup python test.py --num-process 10 --mask-type "orig" --devices $DEVICE --dataset "squadv1" --expl-method "generate_LRP" &
# wait
# nohup python test.py --num-process 10 --mask-type "synt_pos" --devices $DEVICE --dataset "squadv1" --expl-method "generate_LRP" --synt-thres "0.45,0.42,0.75,0.71" &
# wait
# nohup python test.py --num-process 10 --mask-type "upos_pos" --devices $DEVICE --dataset "squadv1" --expl-method "generate_LRP" &
# wait
# nohup python test.py --num-process 10 --devices $DEVICE --dataset "squadv1" --expl-method "AttCAT" &
