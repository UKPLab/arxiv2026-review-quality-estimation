from nlpeer import DATASETS as NLPEER_DATASETS
from nlpeer import PaperReviewDataset

from datetime import datetime

import os
import json
import re
import pandas as pd

import logging
from model.reviewDetails import ReviewDetails
from model.domains import Domains


class ReviewDataset:
    def __init__(self, datasetObj, dataDir):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        self.datasetObj = datasetObj
        self.dataDir = dataDir

        self.reviewDetailsList = []
    
    def getObjectFromDict(self, dataDict):
        return ReviewDetails(
            paperID = dataDict["paperID"],
            paperTitle = dataDict["paperTitle"],
            paperAbstract = dataDict["paperAbstract"],
            paperDecision = dataDict["paperDecision"],
            reviewID = dataDict["reviewID"],
            rawReview = dataDict["rawReview"],
            preprocessedReview = dataDict["preprocessedReview"],
            domain = Domains(dataDict["domain"]),
            datasetName = dataDict["datasetName"],
            date = datetime.fromisoformat(dataDict["date"]),
            scoreDict = dataDict["scoreDict"])
    
class OpenreviewDataset(ReviewDataset):
    def __init__(self, datasetObj, dataDir):
        super().__init__(datasetObj, dataDir)
        
        if len(self.reviewDetailsList) > 0:
            return
        
        # Processed Raw data and save to processed file for future use.
        print(f"Processing raw review data from {os.path.join(self.dataDir, self.datasetObj.value + '.json')}")
        with open(os.path.join(self.dataDir, self.datasetObj.value) + ".json", "r") as f:
            reviewData = json.load(f)
        skipped_paper_count = 0
        paper_count = 0
        review_count = 0
        for paperID, allDetails in reviewData.items():
            if "Decision" not in allDetails or allDetails['Decision'] is None:
                skipped_paper_count += 1
                continue

            # Filter out any papers which have fields missing.
            # TODO: This code is almost never executed. Only once in ICLR-2018 dataset.
            reviewsDict = self._filterIncompleteReviews(allDetails['Reviews'])
            if reviewsDict == {}:
                skipped_paper_count += 1
                continue
            paper_count += 1
            for reviewID, rawReview in reviewsDict.items():
                assert rawReview is not None, f"Review is None for paperID: {paperID}, reviewID: {reviewID}"
                review_count += 1
                self.reviewDetailsList.append(ReviewDetails(paperID = paperID,
                                                        paperTitle = allDetails['Title'],
                                                        paperAbstract = allDetails['Abstract'],
                                                        paperDecision = allDetails['Decision'],
                                                        reviewID = reviewID,
                                                        rawReview = rawReview,
                                                        preprocessedReview = self._reviewToString(datasetObj.value, rawReview),
                                                        domain = Domains.ML,
                                                        datasetName = datasetObj.value,
                                                        date = datetime(int(datasetObj.value.split("-")[1]), 1, 1),
                                                        scoreDict = self._extractAllScores(datasetObj.value, rawReview),
                                                    ))
        print(f"Skipped {skipped_paper_count} papers with no decision or missing reviews")
        print(f"{paper_count} papers and {review_count} reviews loaded from {self.datasetObj.value} dataset")
        
        # Save processed review details to file
        with open(os.path.join(self.dataDir, self.datasetObj.value + "_reviewDetails.json"), "w") as f:
            json.dump([reviewDetail._getDictFromObject() for reviewDetail in self.reviewDetailsList], f, indent=4)
        print(f"Processed review details saved to {os.path.join(self.dataDir, self.datasetObj.value + '_reviewDetails.json')} for future use.")
        
    def _filterIncompleteReviews(self, reviewsDict):
        textFieldsMapping = {
            "ICLR-2018": ["Review"],
            "ICLR-2019": ["Review"],
            "ICLR-2020": ["Review"],
            "ICLR-2021": ["Review"],
            "ICLR-2022": ["Summary Of The Paper", "Main Review", "Summary Of The Review"],
            "ICLR-2023": ["Summary Of The Paper", "Strength And Weaknesses", "Clarity, Quality, Novelty And Reproducibility", "Summary Of The Review"],
            "ICLR-2024": ["Summary", "Strengths", "Weaknesses", "Questions"],
            "ICLR-2025": ["Summary", "Strengths", "Weaknesses", "Questions"],
            "NeurIPS-2021": ["Summary", "Main Review", "Limitations And Societal Impact"],
            "NeurIPS-2022": ["Summary", "Strengths And Weaknesses", "Questions", "Limitations"],
            "NeurIPS-2023": ["Summary", "Strengths", "Weaknesses", "Questions", "Limitations"],
            "NeurIPS-2024": ["Summary", "Strengths", "Weaknesses", "Questions", "Limitations"],
            "NeurIPS-2025": ["Summary", "Strengths And Weaknesses", "Questions", "Limitations", "Paper Formatting Concerns"]
        }

        scoreFieldsMapping = {
            "ICLR-2018": ["Rating", "Confidence"],
            "ICLR-2019": ["Rating", "Confidence"],
            "ICLR-2020": ["Rating", "Experience Assessment"],
            "ICLR-2021": ["Rating", "Confidence"],
            "ICLR-2022": ["Recommendation", "Confidence", "Correctness", "Technical Novelty And Significance", "Empirical Novelty And Significance"],
            "ICLR-2023": ["Recommendation", "Confidence", "Correctness", "Technical Novelty And Significance", "Empirical Novelty And Significance"],
            "ICLR-2024": ["Rating", "Confidence", "Soundness", "Presentation", "Contribution"],
            "ICLR-2025": ["Rating", "Confidence", "Soundness", "Presentation", "Contribution"],
            "NeurIPS-2021": ["Rating", "Confidence"],
            "NeurIPS-2022": ["Rating", "Confidence", "Soundness", "Presentation", "Contribution"],
            "NeurIPS-2023": ["Rating", "Confidence", "Soundness", "Presentation", "Contribution"],
            "NeurIPS-2024": ["Rating", "Confidence", "Soundness", "Presentation", "Contribution"],
            "NeurIPS-2025": ["Rating", "Confidence", "Quality", "Clarity", "Significance", "Originality"]
        }

        filteredReviewsDict = {}
        for reviewID, reviewDict in reviewsDict.items():
            complete = True
            for field in textFieldsMapping[self.datasetObj.value] + scoreFieldsMapping[self.datasetObj.value]:
                if field not in reviewDict:
                    complete = False
                    break
            if complete:
                filteredReviewsDict[reviewID] = reviewDict
        return filteredReviewsDict

    def _reviewToString(self, datasetName, rawReview):
        textFieldsMapping = {
            "ICLR-2018": ["Review"],
            "ICLR-2019": ["Review"],
            "ICLR-2020": ["Review"],
            "ICLR-2021": ["Review"],
            "ICLR-2022": ["Summary Of The Paper", "Main Review", "Summary Of The Review"],
            "ICLR-2023": ["Summary Of The Paper", "Strength And Weaknesses", "Clarity, Quality, Novelty And Reproducibility", "Summary Of The Review"],
            "ICLR-2024": ["Summary", "Strengths", "Weaknesses", "Questions"],
            "ICLR-2025": ["Summary", "Strengths", "Weaknesses", "Questions"],
            "NeurIPS-2021": ["Summary", "Main Review", "Limitations And Societal Impact"],
            "NeurIPS-2022": ["Summary", "Strengths And Weaknesses", "Questions", "Limitations"],
            "NeurIPS-2023": ["Summary", "Strengths", "Weaknesses", "Questions", "Limitations"],
            "NeurIPS-2024": ["Summary", "Strengths", "Weaknesses", "Questions", "Limitations"],
            "NeurIPS-2025": ["Summary", "Strengths And Weaknesses", "Questions", "Limitations", "Paper Formatting Concerns"]
        }
        reviewString = ""
        for field in textFieldsMapping[datasetName]:
            reviewString += f"{field}:\n{rawReview[field]}\n\n"
        return reviewString
    

    def _extractAllScores(self, datasetName, rawReview):
        scoreFieldsMapping = {
            "ICLR-2018": ["Rating", "Confidence"],
            "ICLR-2019": ["Rating", "Confidence"],
            "ICLR-2020": ["Rating", "Experience Assessment"],
            "ICLR-2021": ["Rating", "Confidence"],
            "ICLR-2022": ["Recommendation", "Confidence", "Correctness", "Technical Novelty And Significance", "Empirical Novelty And Significance"],
            "ICLR-2023": ["Recommendation", "Confidence", "Correctness", "Technical Novelty And Significance", "Empirical Novelty And Significance"],
            "ICLR-2024": ["Rating", "Confidence", "Soundness", "Presentation", "Contribution"],
            "ICLR-2025": ["Rating", "Confidence", "Soundness", "Presentation", "Contribution"],
            "NeurIPS-2021": ["Rating", "Confidence"],
            "NeurIPS-2022": ["Rating", "Confidence", "Soundness", "Presentation", "Contribution"],
            "NeurIPS-2023": ["Rating", "Confidence", "Soundness", "Presentation", "Contribution"],
            "NeurIPS-2024": ["Rating", "Confidence", "Soundness", "Presentation", "Contribution"],
            "NeurIPS-2025": ["Rating", "Confidence", "Quality", "Clarity", "Significance", "Originality"]
        }

        scores = {}
        for field in scoreFieldsMapping[datasetName]:
            scores[field] = rawReview[field]
        return scores


class NLPeerReviewDataset(ReviewDataset):
    def __init__(self, datasetObj, dataDir):
        super().__init__(datasetObj, dataDir)

        if len(self.reviewDetailsList) > 0:
            return
        
        print(f"Processing raw review data from NLPeer dataset: {datasetObj.value}")
        
        paperReviewDataset = PaperReviewDataset(dataDir, datasetObj, version=1)
        skipped_paper_count = 0
        paper_count = 0
        review_count = 0
        for paperID, meta, _, reviewsDict in paperReviewDataset:
            if len(reviewsDict) == 0:
                skipped_paper_count += 1
                continue
            
            paper_count += 1
            date = self._extractDate(datasetObj, os.path.join(paperReviewDataset.datapath, paperID, 'meta.json'), meta)

            for idx in reviewsDict.keys() if isinstance(reviewsDict, dict) else range(len(reviewsDict)):
                rawReview = reviewsDict[idx]
                review_count += 1

                self.reviewDetailsList.append(ReviewDetails(paperID = paperID,
                                                        paperTitle = meta['title'],
                                                        paperAbstract = meta['abstract'],
                                                        paperDecision = "Accept",
                                                        reviewID = paperID + "+" + rawReview['rid'],
                                                        rawReview = rawReview,
                                                        preprocessedReview = self._reviewToString(rawReview),
                                                        domain = self._extractDomain(datasetObj, paperID, meta),
                                                        datasetName = datasetObj.value,
                                                        date = date,
                                                        scoreDict = self._extractAllScores(datasetObj.value, rawReview),
                                                    ))
        
        print(f"Skipped {skipped_paper_count} papers with no reviews")
        print(f"{paper_count} papers and {review_count} reviews loaded from {self.datasetObj.value} dataset")
        del paperReviewDataset

        # Save processed review details to file
        with open(os.path.join(self.dataDir, self.datasetObj.value + "_reviewDetails.json"), "w") as f:
            json.dump([reviewDetail._getDictFromObject() for reviewDetail in self.reviewDetailsList], f, indent=4)
        print(f"Processed review details saved to {os.path.join(self.dataDir, self.datasetObj.value + '_reviewDetails.json')} for future use.")

    def _extractDomain(self, datasetObj, paperID, meta):
        if datasetObj in [NLPEER_DATASETS.COLING20, NLPEER_DATASETS.EMNLP23, NLPEER_DATASETS.ACL17, NLPEER_DATASETS.CONLL16, NLPEER_DATASETS.ARR22, NLPEER_DATASETS.ARR_EMNLP_24, NLPEER_DATASETS.ARR_NAACL_25, NLPEER_DATASETS.ARR_ACL_25]:
            return Domains.NLP
        else:
            raise ValueError(f"Dataset {datasetObj.value} not supported yet")


    def _extractDate(self, datasetObj, paperLevelMetaPath, versionLevelMeta):
        if datasetObj == NLPEER_DATASETS.COLING20:
            return datetime(2020, 7, 1)
        elif datasetObj == NLPEER_DATASETS.EMNLP23:
            return datetime(2023, 7, 1)
        elif datasetObj == NLPEER_DATASETS.ACL17:
            return datetime(2017, 3, 1)
        elif datasetObj == NLPEER_DATASETS.CONLL16:
            return datetime(2016, 5, 1)
        elif datasetObj == NLPEER_DATASETS.ARR22:
            assert os.path.exists(paperLevelMetaPath), f"Paper level meta file does not exist: {paperLevelMetaPath}"
            with open(paperLevelMetaPath, 'r') as f:
                paperLevelMeta = json.load(f)
            year = re.findall(r'\d+', paperLevelMeta['cycle'].replace('aclweborgACLARR', ''))[0]
            month = paperLevelMeta['cycle'].replace('aclweborgACLARR', '').replace(year, '')
            return datetime(int(year), datetime.strptime(month, "%B").month, 1)
        elif datasetObj == NLPEER_DATASETS.ARR_EMNLP_24 or datasetObj == NLPEER_DATASETS.ARR_NAACL_25 or datasetObj == NLPEER_DATASETS.ARR_ACL_25:
            assert os.path.exists(paperLevelMetaPath), f"Meta file does not exist: {paperLevelMetaPath}"
            with open(paperLevelMetaPath, 'r') as f:
                paperLevelMeta = json.load(f)
            month, year = paperLevelMeta['cycle'].split('/')[-1], paperLevelMeta['cycle'].split('/')[-2]
            return datetime(int(year), datetime.strptime(month, "%B").month, 1)
        else:
            raise ValueError(f"Unknown dataset: {datasetObj.value}")


    def _reviewToString(self, rawReview):
        reviewString = ""
        for field, value in rawReview['report'].items():
            reviewString += f"{field.replace("_", " ").title()}:\n{value}\n\n"
        return reviewString


    def _extractAllScores(self, datasetName, rawReview):
        scoreDict = rawReview['scores'].copy()
        
        # Pop best paper
        scoreDict.pop("best_paper", None)

        # Add confidence
        scoreDict["confidence"] = rawReview['meta']['confidence']

        if datasetName == "ARR-22":
            # Remove text from fields, convert scores to int and rename "overall" key to "overall_assessment" to be consistent with other datasets
            for scoreKey, scoreValue in scoreDict.items():
                scoreDict[scoreKey] = float(scoreValue.split(" = ")[0])
            scoreDict["overall_assessment"] = scoreDict["overall"]
            scoreDict.pop("overall")
        
        return scoreDict
