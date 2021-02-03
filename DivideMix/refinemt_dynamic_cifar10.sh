#!/bin/bash
# python Train_cifar.py --gpuid 1 --noise_mode asym --r 0.4 --distill dynamic --refinement --lambda_u 0
python Train_cifar.py --gpuid 1 --noise_mode sym --r 0.2 --distill dynamic --refinement --lambda_u 0 --distill_mode kmeans
python Train_cifar.py --gpuid 1 --noise_mode sym --r 0.8 --distill dynamic --refinement --lambda_u 25 --distill_mode kmeans
