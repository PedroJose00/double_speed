#!/bin/bash
 
#SBATCH --partition=normal
#SBATCH --job-name=double_speed_20_05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:1
#SBATCH -o test_%j.out
#SBATCH -e test_%j.err
 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pjdiaz@uis.edu.co

module load Analytics/Anaconda/python3
source activate fenicsx-env

python double_speed_20_05_24.py
