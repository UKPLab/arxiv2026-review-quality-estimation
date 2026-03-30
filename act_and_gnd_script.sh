#!/bin/bash
#
#SBATCH --job-name=act_and_gnd_script
#SBATCH --output=logs/act_and_gnd_script_%j.out
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --account=
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_model:a100"

# Create environment from https://github.com/bodasadallah/RevUtil and activate it
source ...

# Replace venue name as needed
VENUE="ARR2022"

# Run the script to calculate the metrics for all the datasets and venues
PREPROCESSED_INPUT_FILE_PATH=$PWD/rqe_data/sample/preprocessed/$VENUE.json

python $PWD/utils/act_and_gnd/act_and_gnd_calculator.py --input_file_path $PREPROCESSED_INPUT_FILE_PATH --output_file_dir $PWD/rqe_data/precomputed_metrics/m_act_gnd/