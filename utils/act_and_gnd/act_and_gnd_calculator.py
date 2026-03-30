import argparse
import os
import json

from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
from vllm.lora.request import LoRARequest
from allCodeFromRevUtil import get_prompt, extract_predictions

def parse_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--input_file_path", type=str, required=True)
    parser.add_argument("--output_file_dir", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Read data and create prompts for each review point
    dataFilePath = os.path.join(args.input_file_path)
    with open(dataFilePath, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")
    allReviewPoints = []
    allReviewIDs = []

    for reviewDetails in data:
        reviewID = reviewDetails['reviewID']
        assert 'segmentedReview' in reviewDetails, f"Please pass itemized review data"
        reviewPoints = reviewDetails['segmentedReview']['weaknesses_suggestions_questions_comments_segments']
        allReviewPoints.extend(reviewPoints)
        allReviewIDs.extend([reviewID]*len(reviewPoints))

    prompts = [get_prompt(reviewPoint) for reviewPoint in allReviewPoints]
    assert len(prompts) == len(allReviewPoints) == len(allReviewIDs), "Mismatch in lengths of prompts, review points and review IDs"
    print(f"Created {len(prompts)} prompts for WEAKNESSES section")


    # Initialize model
    score_lora_path = snapshot_download(repo_id="boda/RevUtil_Llama-3.1-8B-Instruct_score_only")
    scoreLoraRequest = LoRARequest("revutil_adapter", 1, score_lora_path)

    rationale_lora_path = snapshot_download(repo_id="boda/RevUtil_Llama-3.1-8B-Instruct_score_rationale")
    rationaleLoraRequest = LoRARequest("revutil_adapter", 1, rationale_lora_path)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

    llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", enable_lora=True, max_lora_rank=64, tensor_parallel_size=1, gpu_memory_utilization=0.95, enable_prefix_caching=True, max_model_len = 8196, trust_remote_code=True)

    print("Model initialization done!")


    # Start inference
    outputFilePath = os.path.join(args.output_file_dir, args.input_file_path.split("/")[-1])
    print(f"Starting inference for WEAKNESSES section, outputs will be saved to {outputFilePath}")
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=True, lora_request=scoreLoraRequest)
    predictions = extract_predictions(outputs)

    print(f"Inference completed, saving output")
    ## Put in desired format
    allOutputs = []
    cntr = 0
    for reviewDetails in data:
        reviewID = reviewDetails['reviewID']
        reviewPoints = reviewDetails['segmentedReview']['weaknesses_suggestions_questions_comments_segments']
        predictionForReview = []
        for idx in range(cntr, cntr + len(reviewPoints)):
            assert allReviewIDs[idx] == reviewID, "Mismatch in review IDs"
            assert allReviewPoints[idx] == reviewPoints[idx - cntr], "Mismatch in review points"
            prediction = predictions[idx]
            predictionForReview.append(prediction)
        cntr += len(reviewPoints)
        allOutputs.append({
            "reviewID": reviewID,
            "segmentsProcessed": reviewPoints,
            "predictedScores": predictionForReview
        })
    assert cntr == len(predictions), "Mismatch in total predictions count"

    ## Save predictions
    with open(outputFilePath, "w") as f:
        json.dump(allOutputs, f, indent=4)

