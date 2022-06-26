#!/bin/bash
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=128000M
#SBATCH --cpus-per-task=32
#SBATCH --time=15:00:00
#SBATCH --account=def-zaiane
source ~/ENV/bin/activate
python project_690_v4_boosting_try20copy4_oldsig.py