#!/bin/bash
#
#SBATCH -p short
#SBATCH --export=ALL
#SBATCH --job-name=intra_option5
#SBATCH --output=log3.txt
#
#SBATCH --array=0-1000%100
#SBATCH --time=60:00

cd $HOME/Code/fourier/fourier
EXPID=$((SLURM_ARRAY_TASK_ID + 5000)) 
python run_exp.py -id $EXPID
