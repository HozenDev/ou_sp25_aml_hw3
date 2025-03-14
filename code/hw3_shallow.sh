#!/bin/bash
#
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=30G
#SBATCH --output=results/hw3_shallow_%j_stdout.txt
#SBATCH --error=results/hw3_shallow_%j_stderr.txt
#SBATCH --time=02:00:00
#SBATCH --job-name=hw3_shallow
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/hw3/code/
#SBATCH --array=0

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh
conda activate dnn

## SHALLOW
python hw3_base.py -v @exp.txt @oscer.txt @net_shallow.txt --exp_index $SLURM_ARRAY_TASK_ID --cpus_per_task $SLURM_CPUS_PER_TASK --epochs 3000 -vvv  --cache "" --save_model --render --nogo
