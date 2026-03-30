#!/bin/bash
#
#SBATCH --job-name=preproc_script
#SBATCH --output=logs/preproc_script_%j.out
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --account=
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1


# Activate environment created based on requirements.txt in this repo
source ...

# Replace venue name as needed
VENUE="ARR2022"

# Run the script to preprocess data
python $PWD/utils/preproc/data_preprocessor.py --input_file_path $PWD/rqe_data/full/$VENUE.json --output_file_dir $PWD/rqe_data/sample/preprocessed/
