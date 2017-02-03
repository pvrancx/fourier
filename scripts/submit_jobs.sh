#!/bin/bash
#
#SBATCH --job-name=intra_option
#SBATCH --output=$HOME/Code/fourier/log.txt
#
#SBATCH --ntasks=1
#SBATCH --time=60:00

cd $HOME/Code/fourier/fourier
python run_exp.py -id $SLURM_PROCID
