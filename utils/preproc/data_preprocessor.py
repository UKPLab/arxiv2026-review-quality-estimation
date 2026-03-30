from tqdm import tqdm
import argparse
import os
from collections import defaultdict
import json

import re
import spacy
from spacy.symbols import ORTH
from nlpeer.data.create.utils import clean_and_split_review

from prompts import StructurerPrompts

from openai import OpenAI

from pydantic import BaseModel

class SegmentedReview(BaseModel):
    summary_segments: list[str]
    strengths_segments: list[str]
    weaknesses_suggestions_questions_comments_segments: list[str]
    other_segments: list[str]

def parse_args():
    parser = argparse.ArgumentParser()

    # Takes the path of the input file and the output directory as arguments. The output file will be saved in the output directory with the same name as the input file.
    parser.add_argument("--input_file_path", type=str, required=True)
    parser.add_argument("--output_file_dir", type=str, required=True)
    
    return parser.parse_args()


def checkFields(reviewData):
    required_fields = ["paperID", "paperDecision", "reviewID", "domain", "datasetName", "date", "rawReview", "preprocessedReview", "scoreDict", "paperTitle", "paperAbstract"]
    for review in reviewData:
        for field in required_fields:
            if field not in review:
                raise ValueError(f"Missing field {field} in review {review.get('reviewID', 'unknown')}")
    print("All required fields are present.")


# Sentence splitting related code

# Taken from https://github.com/UKPLab/nlpeer/blob/832352c33f874b4fd207792313ac4916a6137b52/src/nlpeer/data/create/utils.py#L69-L129

nlp = spacy.load('en_core_sci_sm', exclude=["ner", "tagger", "parser", "lemmatizer"])
nlp.add_pipe("sentencizer")

nlp.tokenizer.add_special_case("<br>", [{ORTH: "<br>"}])
nlp.add_pipe("linebreak_component", name="linebreaking", first=True)

def augmented_clean(txt):
    res = txt.strip()  # strip unnecessary whitespaces

    res = re.sub(r"\n{2,}", " <br> ", res)  # clear line break
    res = re.sub(r" <br> [*\-] ", " <br> - ", res)  # replaced line break with an itemize
    res = re.sub(r"\n[*\-] ", " <br> - ", res)  # non-replaced line break with an itemize
    res = re.sub(r"^\* ", "- ", res)

    return res

def sentence_splitter_from_NLPeer(text):
    cleaned = augmented_clean(text)
    processed = nlp(cleaned)

    # get sentences and replace the <br> parts by actual line breaks
    new_text = [s.text for s in processed.sents]
    for i, t in enumerate(new_text):
        if t == "<br>" and i > 0:
            new_text[i-1] = new_text[i-1] + "\n"

    # update sentences to exclude <br> ones and replace any br left in a sentence
    new_text = [t.replace("<br>", "") for t in new_text if t != "<br>"]

    # if line breaks do not occur at the beginning or ending of a sentence, they should be discarded
    tmp = []
    for t in new_text:
        if len(t) > 1:
            e = t[0]+ t[1:-1].replace("\n", " ") + t[-1]
        else:
            e = t

        if e[-1] not in [" ", "\n", "\t"]:
            e += " "

        tmp += [e]

    new_text = tmp

    sentences = []
    ix = 0
    for s in new_text:
        sentences += [(ix, ix+len(s))]
        ix += len(s)

    # Altered here to return sentences and not just the indices
    return new_text


def get_itemized_review_GPT(client, review_text):
    # Initialize Prompts
    promptDict = StructurerPrompts().promptDict

    systemPrompt = promptDict["system"]
    userPrompt = promptDict["user"]
    
    messages = [
        { "role": "system", "content": systemPrompt },
        { "role": "user", "content": userPrompt.format(examples=StructurerPrompts().examples, review_text=review_text) }
    ]

    result = client.responses.parse(
            model = "gpt-5.2",
            input = messages,
            reasoning={ "effort": "medium" },
            text={ "verbosity": "low" },
            text_format = SegmentedReview
        ).output_parsed
    
    return result


if __name__ == "__main__":
    args = parse_args()

    with open(args.input_file_path, "r") as f:
        reviewData = json.load(f)
    
    # Check if all fields are available
    checkFields(reviewData)

    # Initialize OpenAI client for itemization
    client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )

    # process reviews and save file
    print("Starting to process reviews...")
    for review in tqdm(reviewData):
        review["sentences"] = sentence_splitter_from_NLPeer(review["preprocessedReview"])
        review["segmentedReview"] = get_itemized_review_GPT(client, review["preprocessedReview"]).model_dump(mode="json")
    print("Processed all reviews. Saving to file...")

    # save the processed reviews
    output_path = os.path.join(args.output_file_dir, args.input_file_path.split("/")[-1])
    with open(output_path, "w") as f:
        json.dump(reviewData, f, indent=4)
    print(f"Saved processed reviews to {output_path}")