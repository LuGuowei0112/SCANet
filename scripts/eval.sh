#!/usr/bin/env bash

gpus=0

data_name=LEVIR
net_G=SCANet
split=test
project_name=CD_SCANet_LEVIR_b8_lr0.01_train_val_200_linear
checkpoint_name=best_ckpt.pt

python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


