#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=30G
#SBATCH --output=results/hw3_shallow_%j_stdout.txt
#SBATCH --error=results/hw3_shallow_%j_stderr.txt
#SBATCH --time=02:00:00
#SBATCH --job-name=hw3_shallow
#SBATCH --mail-user=ADD YOUR OWN EMAIL
#SBATCH --mail-type=ALL
#SBATCH --chdir=ADD YOUR OWN CHDIR
##SBATCH --array=0-4    # the double ## means that this line is ignored

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh
conda activate dnn

## SHALLOW
python hw3_base.py -v @exp.txt @oscer.txt @net_shallow.txt --exp_index $SLURM_ARRAY_TASK_ID --cpus_per_task $SLURM_CPUS_PER_TASK --epochs 3000 -vvv  --cache "" --save_model --render
