#!/bin/bash
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=128000M
#SBATCH --cpus-per-task=32
#SBATCH --time=15:00:00
#SBATCH --account=def-zaiane
source ~/ENV/bin/activate
python cmd2_copy4.py ionosphere