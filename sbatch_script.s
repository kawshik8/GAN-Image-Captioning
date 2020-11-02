#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:1
#SBATCH --time=6:00:00
#SBATCH --mem=50000
#SBATCH --job-name=kk4161
#SBATCH --mail-user=kk4161@nyu.edu
#SBATCH --output=slurm_%j.out

#source ../../env/bin/activate
module load python3/intel/3.5.3
source ../../env/bin/activate
#pip3 install tensorboardX

python3 src/main.py --data-dir ../coco_data --save-dir save --pretrain-epochs 0 --pretrain-lr 1e-3 --gen-lr 1e-4 --disc-lr 1e-4 --dataset_percent 0.1 --expt-name adv_debug --captions-per-image 5
