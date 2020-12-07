#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:1
#SBATCH --time=24:00:00
#SBATCH --mem=50000
#SBATCH --job-name=kk4161
#SBATCH --mail-user=kk4161@nyu.edu
#SBATCH --output=slurm_%j.out

#. ~/.bashrc
#module load anaconda3
#module load cudnn/10.1v7.6.5.32
#module load cuda/10.1.105

#conda activate pytorch
#conda install -n pytorch nb_conda_kernels

######## pretrain ###########
#python3 src/main.py --gen-embed-dim 256 --gen-hidden-dim 256 --gen-num-layers 1 --gen-model-type lstm --freeze-cnn 0 --num-workers 8 --data-dir ../coco_data --save-dir save --pretrain-lr 1e-4 --pretrain-epochs 100 --gen-lr 1e-4 --disc-lr 1e-4 --adv-epochs 0 --dataset_percent 1.0 --expt-name lstm-256-unconditional --captions-per-image 5 --pretrain-patience 12 --advtrain-patience 12 --pretrain-lr-patience 4 --conditional-gan 0

######## advtrain ###########
####### LSTM with trans disc
#python3 src/main.py --gen-num-layers 1 --gen-model-type lstm --freeze-cnn 1 --num-workers 8 --data-dir ../coco_data --save-dir save --pretrain-lr 1e-4 --pretrain-epochs 0 --gen-lr 1e-4 --disc-lr 1e-4 --adv-epochs 100 --dataset_percent 0.1 --expt-name lstm-1StepGenTrain-disc-trans --disc-num-layers 2 --disc-type transformer --captions-per-image 5 --advtrain-patience 200 --pretrain-lr-patience 8 --pretrain-patience 25 --conditional-gan 0 --adv-train-batch-size 32 --adv-loss-type standard 

####### LSTM with cnn disc
python3 src/main.py --gen-num-layers 1 --gen-model-type lstm --freeze-cnn 1 --num-workers 8 --data-dir ../coco_data --save-dir save --pretrain-lr 1e-4 --pretrain-epochs 0 --gen-lr 1e-4 --disc-lr 1e-4 --adv-epochs 200 --dataset_percent 0.1 --expt-name lstm-1StepGenTrain-cnn-disc-flipped-labels --flip-labels 1 --gen-steps 1 --disc-type cnn --captions-per-image 5 --advtrain-patience 200 --pretrain-lr-patience 8 --pretrain-patience 25 --conditional-gan 0 --adv-train-batch-size 32 --adv-loss-type standard

######## Transformer with cnn disc
#python3 src/main.py --gen-num-layers 3 --gen-model-type transformer --freeze-cnn 0 --num-workers 8 --data-dir ../coco_data --save-dir save --pretrain-lr 1e-4 --pretrain-epochs 0 --gen-lr 1e-4 --disc-lr 5e-4 --adv-epochs 200 --dataset_percent 0.1 --expt-name transformer-3l-2StepGenTrain --captions-per-image 5 --advtrain-patience 200 --pretrain-lr-patience 4 --conditional-gan 0 --adv-train-batch-size 16 --adv-loss-type standard

######## Transformer with trans disc
#python3 src/main.py --gen-num-layers 3 --gen-model-type transformer --freeze-cnn 0 --num-workers 8 --data-dir ../coco_data --save-dir save --pretrain-lr 1e-4 --pretrain-epochs 0 --gen-lr 1e-4 --disc-lr 5e-4 --adv-epochs 200 --dataset_percent 0.1 --expt-name transformer-3l-2StepGenTrain-disc-trans --captions-per-image 5 --disc-type transformer --disc-num-layers 2 --advtrain-patience 200 --pretrain-lr-patience 4 --conditional-gan 0 --adv-train-batch-size 16 --adv-loss-type standard
########3 pretrain - adv #########
#python3 src/main.py --num-workers 8 --data-dir ../coco_data --save-dir save --pretrain-lr 1e-4 --pretrain-epochs 0 --gen-lr 1e-4 --disc-lr 1e-4 --adv-epochs 100 --dataset_percent 0.1 --expt-name pre_adv_debug --captions-per-image 5 --pretrain-patience 25 --advtrain-patience 25
#python3 src/main.py --gen-embed-dim 32 --gen-hidden-dim 256 --gen-num-layers 1 --gen-model-type lstm --num-workers 8 --data-dir ../coco_data --save-dir save --pretrain-lr 1e-4 --pretrain-epochs 100 --gen-lr 1e-4 --disc-lr 1e-4 --adv-epochs 100 --dataset_percent 0.1 --expt-name pre_adv_debug --captions-per-image 5 --pretrain-patience 25 --advtrain-patience 25
