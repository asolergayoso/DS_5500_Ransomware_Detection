#!/bin/bash
#SBATCH --job-name=feature_engineering
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --output=/home/mali.om/capstone/logs/feature_eng_%j.out
#SBATCH --error=/home/mali.om/capstone/logs/feature_eng_%j.err
#SBATCH --mail-user=mali.om@northeastern.edu
#SBATCH --mail-type=ALL

DATA_DIR=""
DEST_DIR=""
export LOG_PATH=""
export SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

#conda env
source activate 

#code
python /home/mali.om/parallel_feature_engineering.py --data-dir $DATA_DIR --output-dir $DEST_DIR --window-size 10 --step-size 1