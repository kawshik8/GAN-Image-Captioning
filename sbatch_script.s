#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:1
#SBATCH --time=6:00:00
#SBATCH --mem=50000
#SBATCH --job-name=kk4161
#SBATCH --mail-user=kk4161@nyu.edu
#SBATCH --output=slurm_%j.out

#source ../../env/bin/activate
#module load python3/intel/3.5.3
#source ../../env/bin/activate
#pip3 install tensorboardX
#conda init
#conda activate myenv
#pip3 install tensorboardX

python3 src/main.py --num-workers 8 --data-dir ../coco_data --save-dir save --pretrain-lr 1e-3 --gen-lr 1e-4 --disc-lr 1e-4 --dataset_percent 0.1 --expt-name adv_debug --captions-per-image 5
