import argparse
import os
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--orig_input_file_path", type=str, required=True, help="Location where the original input file is saved")
    parser.add_argument("--temp_input_file_dir", type=str, required=True, help="Location where the input file is saved")
    parser.add_argument("--temp_output_file_dir", type=str, required=True, help="Location where predicted output is saved")

    parser.add_argument("--output_file_dir", type=str, required=True, help="Location where the output files will be saved")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Read model predictions
    predFilePath = os.path.join(args.temp_output_file_dir, f"predicted.json")
    with open(predFilePath, "r") as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions from {predFilePath}")

    # Read input data
    inputFilePath = os.path.join(args.temp_input_file_dir, f"input.json")
    with open(inputFilePath, "r") as f:
        inputData = json.load(f)
    print(f"Loaded input data from {inputFilePath}")

    # Convert to Pragtag output format
    pragtagOutputDataList = []
    for i in range(len(inputData)):
        assert inputData[i]['id'] == predictions[i]['id'], "Mismatch between input data and predictions"
        outputData = {}
        outputData['reviewID'] = inputData[i]['id']
        outputData['sentences'] = inputData[i]['sentences']
        outputData['labels'] = predictions[i]['labels']

        pragtagOutputDataList.append(outputData)
    
    outputFilePath = os.path.join(args.output_file_dir, args.orig_input_file_path.split("/")[-1])
    with open(outputFilePath, "w") as f:
        json.dump(pragtagOutputDataList, f, indent=4)
    print(f"Saved Pragtag output data to {outputFilePath}")
    