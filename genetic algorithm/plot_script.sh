#!/bin/bash
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH --mem-per-cpu=2000

module load anaconda3
source activate multiobj-guided-evac

python plot_fronts.py
