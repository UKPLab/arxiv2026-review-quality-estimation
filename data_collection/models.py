from nlpeer import DATASETS as NLPEER_DATASETS
from enum import Enum
import json

class OPENREVIEW_DATASETS(Enum):
    ICLR_18 = "ICLR-2018"
    ICLR_19 = "ICLR-2019"
    ICLR_20 = "ICLR-2020"
    ICLR_21 = "ICLR-2021"
    ICLR_22 = "ICLR-2022"
    ICLR_23 = "ICLR-2023"
    ICLR_24 = "ICLR-2024"
    ICLR_25 = "ICLR-2025"
    NEURIPS_21 = "NeurIPS-2021"
    NEURIPS_22 = "NeurIPS-2022"
    NEURIPS_23 = "NeurIPS-2023"
    NEURIPS_24 = "NeurIPS-2024"
    NEURIPS_25 = "NeurIPS-2025"

class Domains(Enum):
    NLP = "NLP"
    Multi = "MULTI-DOMAIN"
    ML = "ML"

class ReviewDetails:
    def __init__(self, paperID, paperTitle, paperAbstract, paperDecision, reviewID, rawReview, preprocessedReview, domain, datasetName, date, scoreDict):
        self.paperID = paperID
        self.paperTitle = paperTitle
        self.paperAbstract = paperAbstract
        self.paperDecision = paperDecision
        self.reviewID = reviewID
        self.rawReview = rawReview
        self.preprocessedReview = preprocessedReview
        self.domain = domain
        self.datasetName = datasetName
        self.date = date
        self.scoreDict = scoreDict

    def _getDictFromObject(self):
        return {
            "paperID": self.paperID,
            "paperTitle": self.paperTitle,
            "paperAbstract": self.paperAbstract,
            "paperDecision": self.paperDecision,
            "reviewID": self.reviewID,
            "rawReview": self.rawReview,
            "preprocessedReview": self.preprocessedReview,
            "domain": self.domain.value,
            "datasetName": self.datasetName,
            "date": self.date.isoformat(),
            "scoreDict": self.scoreDict
        }

    def __repr__(self):
        return json.dumps(self._getDictFromObject(), indent=4)

