import argparse
import os
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file_path", type=str, required=True)
    parser.add_argument("--output_file_dir", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Read data
    dataFilePath = os.path.join(args.input_file_path)
    with open(dataFilePath, "r") as f:
        sampledReviewsWithSentences = json.load(f)

    # Make a new file based on the input format used by Pragtag model
    # Each entry is a dict with 4 keys: 'id', 'sentences', 'pid' and 'domain'
    pragtagInputDataList = []
    for sampledReview in sampledReviewsWithSentences:
        inputData = {}
        inputData['id'] = sampledReview['reviewID']
        inputData['sentences'] = sampledReview['sentences']
        inputData['pid'] = sampledReview['paperID']
        inputData['domain'] = sampledReview['domain']

        pragtagInputDataList.append(inputData)
    
    outputFilePath = os.path.join(args.output_file_dir, f"input.json")
    with open(outputFilePath, "w") as f:
        json.dump(pragtagInputDataList, f, indent=4)