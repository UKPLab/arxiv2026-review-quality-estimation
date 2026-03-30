#!/bin/bash
#
#SBATCH --job-name=req_script
#SBATCH --output=logs/req_script_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --account=
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_model:a100"


# Install environment from https://github.com/UKPLab/pragtag2023/tree/main and activate it
source ...

# Run script to preprocess input before running the model
INPUT_TEMP_DIR=$PWD/utils/req/temp_preprocessed_input
OUTPUT_TEMP_DIR=$PWD/utils/req/temp_predicted_output

# Create temp directories if they don't exist
mkdir -p $INPUT_TEMP_DIR
mkdir -p $OUTPUT_TEMP_DIR

# Add the path of the Pragtag model from https://github.com/UKPLab/pragtag2023/tree/main
PRAGTAG_MODEL_PATH=...

# Replace venue name as needed
VENUE="ARR2022"

PREPROCESSED_INPUT_FILE_PATH=$PWD/rqe_data/sample/preprocessed/$VENUE.json

python $PWD/utils/req/input_maker.py --input_file_path $PREPROCESSED_INPUT_FILE_PATH --output_file_dir $INPUT_TEMP_DIR

# run the PragTag model
python $PWD/utils/req/predict_baseline.py $INPUT_TEMP_DIR/input.json $PRAGTAG_MODEL_PATH $OUTPUT_TEMP_DIR

# Save the output in original repo
python $PWD/utils/req/output_maker.py --orig_input_file_path $PREPROCESSED_INPUT_FILE_PATH --temp_input_file_dir $INPUT_TEMP_DIR --temp_output_file_dir $OUTPUT_TEMP_DIR --output_file_dir $PWD/rqe_data/precomputed_metrics/m_req/

# Remove temp directories
rm -rf $INPUT_TEMP_DIR
rm -rf $OUTPUT_TEMP_DIR