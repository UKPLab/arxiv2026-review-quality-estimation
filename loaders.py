import json, os, tqdm, collections, logging
import pandas as pd

# how do we decide what's "Accepted"
PAPER_DECISION_MAP = {"Reject": "Reject",
                      "Accept - Poster": "Accept",
                      "Accept (poster)": "Accept",
                      "Accept (Poster)": "Accept",
                      "Accept": "Accept",
                      "Accept (spotlight)": "Accept",
                      "Accept: poster": "Accept",
                      "Accept (Spotlight)": "Accept",
                      "Accept (Oral)": "Accept",
                      "Accept (oral)": "Accept",
                      "Accept - Oral": "Accept",
                      "Accept: notable-top-25%": "Accept",
                      "Accept: notable-top-5%": "Accept",
                      "Invite to Workshop Track": "Accept",  # treat workshop track as accept (ICLR-2018 only)
                      "Accept (Talk)": "Accept",
                      "Accept - Shepherd": "Accept",
                      "Withdrawn": "Withdrawn",
                      # most withdrawn and desk-rejected papers filtered out during crawl, tidy up the rest
                      "Desk Rejected": "Desk Reject"}


# Load review data into a dataframe (works for full and sample data)
def make_dataframe(data_dir):
    data = []
    for fn in tqdm.tqdm(os.listdir(data_dir), desc=f"Loading review data from {data_dir}"):
        if fn.endswith(".json"):
            c = os.path.splitext(fn)[0]
            venue, year = c[:-4], int(c[-4:])  # last 4 chars are year
            campaign = f"{venue}-{year}"
            with open(os.path.join(data_dir, fn)) as f:
                for r in json.load(f):
                    review = {
                        "venue": venue,
                        "year": year,
                        "campaign": campaign,
                        "reviewID": r["reviewID"],
                        "paperID": r["paperID"],
                        "paperTitle": r["paperTitle"],
                        "paperDecision": r["paperDecision"],
                        "paperDecisionCoarse": PAPER_DECISION_MAP[r["paperDecision"]],
                        "flattenedReview": r["preprocessedReview"],
                    }
                    if "segmentedReview" in r:
                        for seg in r["segmentedReview"]:
                            review[f"seg_{seg.replace('_segments', '')}"] = len(r["segmentedReview"][seg])

                    # Add quality score if available (only available in ACL-2018)
                    if "qualityScore" in r:
                        review["qualityScore"] = r["qualityScore"]
                        review["overallScore"] = r["rawReview"]["scores"]["overall_score"]
                        review["confidenceScore"] = r["rawReview"]["confidence"]

                    data += [review]
    return pd.DataFrame(data)


# Load author utility metrics (ACT, GND...) from predictions into a dataframe
def read_au_metrics(au_src_dir):
    data = []
    no_claim = collections.defaultdict(int)

    for fn in tqdm.tqdm(os.listdir(au_src_dir), desc="Loading AU metrics"):
        if fn.endswith(".json"):
            with open(os.path.join(au_src_dir, fn)) as f:
                for r in json.load(f):
                    review = {"reviewID": r["reviewID"]}
                    scores = {"act": [], "gnd": [], "ver": []}
                    for ss in r["predictedScores"]:  # score set per segment
                        if ss["actionability_label"] != "X":  # 'X' means no claim detected
                            scores["act"] += [int(ss["actionability_label"])]  # note, original label is categorical
                        if ss["grounding_specificity_label"] != "X":
                            scores["gnd"] += [int(ss["grounding_specificity_label"])]
                        if ss["verifiability_label"] != "X":
                            scores["ver"] += [int(ss["verifiability_label"])]
                    #  AU score averaging happens here
                    for s in scores:
                        if len(scores[s]) == 0:
                            no_claim[s] += 1
                            scores[s] = 0
                        else:
                            scores[s] = sum(scores[s]) / len(scores[s])

                    review["ACT"] = scores["act"]
                    review["GND"] = scores["gnd"]
                    # review["VER"] = scores["ver"]
                    data += [review]
    for s in no_claim:
        logging.warning(f"No [{s}] scores in {no_claim[s]} reviews, assigned 0")
    return pd.DataFrame(data)

# Load REQ metric from predictions into a dataframe
def read_rq_metric(rq_src_dir):
    data = []
    for fn in tqdm.tqdm(os.listdir(rq_src_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(rq_src_dir, fn)) as f:
                for r in json.load(f):
                    review = {
                        "reviewID": r["reviewID"],
                        "REQ": r["labels"].count("Todo")
                    }
                    data += [review]
    return pd.DataFrame(data)