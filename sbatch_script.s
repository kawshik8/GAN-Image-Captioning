#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --time=24:00:00
#SBATCH --mem=50000
#SBATCH --job-name=su575
#SBATCH --mail-user=su575@nyu.edu
#SBATCH --output=slurm_%j_p40.out

. ~/.bashrc
module load anaconda3
module load cudnn/10.1v7.6.5.32
module load cuda/10.1.105

conda activate pytorch
conda install -n pytorch nb_conda_kernels

######## pretrain ###########
#python3 src/main.py --gen-embed-dim 256 --gen-hidden-dim 256 --gen-num-layers 1 --gen-model-type lstm --freeze-cnn 0 --num-workers 8 --data-dir ../coco_data --save-dir save --pretrain-lr 1e-4 --pretrain-epochs 100 --gen-lr 1e-4 --disc-lr 1e-4 --adv-epochs 0 --dataset_percent 1.0 --expt-name lstm-256-unconditional --captions-per-image 5 --pretrain-patience 12 --advtrain-patience 12 --pretrain-lr-patience 4 --conditional-gan 0

######## advtrain ###########
python3 src/main.py --gen-num-layers 1 --gen-model-type lstm --freeze-cnn 0 --num-workers 8 --data-dir ../coco_data --save-dir save --pretrain-lr 1e-4 --pretrain-epochs 0 --gen-lr 1e-4 --disc-lr 1e-4 --adv-epochs 100 --dataset_percent 0.1 --expt-name lstm-2StepGenTrain --captions-per-image 5 --advtrain-patience 12 --pretrain-lr-patience 4 --conditional-gan 0 --adv-train-batch-size 32 --adv-loss-type standard 
########3 pretrain - adv #########
#python3 src/main.py --num-workers 8 --data-dir ../coco_data --save-dir save --pretrain-lr 1e-4 --pretrain-epochs 0 --gen-lr 1e-4 --disc-lr 1e-4 --adv-epochs 100 --dataset_percent 0.1 --expt-name pre_adv_debug --captions-per-image 5 --pretrain-patience 25 --advtrain-patience 25
#python3 src/main.py --gen-embed-dim 32 --gen-hidden-dim 256 --gen-num-layers 1 --gen-model-type lstm --num-workers 8 --data-dir ../coco_data --save-dir save --pretrain-lr 1e-4 --pretrain-epochs 100 --gen-lr 1e-4 --disc-lr 1e-4 --adv-epochs 100 --dataset_percent 0.1 --expt-name pre_adv_debug --captions-per-image 5 --pretrain-patience 25 --advtrain-patience 25
