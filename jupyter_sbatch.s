#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --mem=10000
#SBATCH --job-name=kk4161
#SBATCH --mail-user=kk4161@nyu.edu
#SBATCH --output=slurm_%j.out


. ~/.bashrc
module load anaconda3/5.3.1
module load jupyter-kernels/py3.5

conda activate ../../env/bin/activate

port=$(shuf -i 10000-65500 -n 1)
/usr/bin/ssh -N -f -R $port:localhost:$port log-0
/usr/bin/ssh -N -f -R $port:localhost:$port log-1

cat<<EOF

Jupyter server is running on: $(hostname)
Job starts at: $(date)

Step 1 :
ssh -L $port:localhost:$port $USER@prince.hpc.nyu.edu

ssh -L $port:localhost:$port $USER@prince

Step 2:


the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)
