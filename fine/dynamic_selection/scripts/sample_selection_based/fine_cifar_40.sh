#!/bin/bash

python main.py -d 1 --asym false --percent 0.4 --lr_scheduler multistep --arch rn18 --loss_fn cce --dataset cifar10 --traintools robustloss --no_wandb --dynamic --distill_mode fine-gmm --seed 123 --warmup 40