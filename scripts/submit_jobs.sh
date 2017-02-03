#!/bin/bash
#
#SBATCH -p short
#SBATCH --export=ALL
#SBATCH --job-name=intra_option
#SBATCH --output=$HOME/Code/fourier/log.txt
#
#SBATCH sbatch --array=0-1
#SBATCH --time=60:00

cd $HOME/Code/fourier/fourier
python run_exp.py -id $SLURM_ARRAY_TASK_ID
