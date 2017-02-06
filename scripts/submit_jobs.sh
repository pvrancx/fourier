#!/bin/bash
#
#SBATCH -p short
#SBATCH --export=ALL
#SBATCH --job-name=intra_option_bias
#SBATCH --output=log.txt
#
#SBATCH --array=0-500
#SBATCH --time=60:00

cd $HOME/Code/fourier/fourier
EXPID=$((SLURM_ARRAY_TASK_ID))
python run_exp.py -id $EXPID ../scripts/cfg.txt
