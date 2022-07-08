#!/usr/bin/bash
!python train_bert.py --gpu \
                     --batch_size 32 \
                     --lr 6e-6 \
                     --epoch 30 \
                     --data_dir ../data/ \
                     --bert_model_dir ../model/chinese_wwm_ext_pytorch/ \
                     --model_save_path ../model/best_bert_model_V2


