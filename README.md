# Is Peer Review Really in Decline? Analyzing Review Quality across Venues and Time

[![Arxiv](https://img.shields.io/badge/Arxiv-2601.15172-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.15172)
[![License](https://img.shields.io/badge/License-Apache--2.0-green?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.12-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

This repository contains all scripts and data-links necessary for reproducing the results for estimating quality of reviews.

---

> **Abstract:** Peer review is at the heart of modern science. As submission numbers rise and research communities grow, the decline in review quality is a popular narrative and a common concern. Yet, is it true? Review quality is difficult to measure, and the ongoing evolution of reviewing practices makes it hard to compare reviews across venues and time. To address this, we introduce a new framework for evidence-based comparative study of review quality and apply it to major AI and machine learning conferences: ICLR, NeurIPS and *ACL. We document the diversity of review formats and introduce a new approach to review standardization. We propose a multi-dimensional schema for quantifying review quality as utility to editors and authors, coupled with both LLM-based and lightweight measurements. We study the relationships between measurements of review quality, and its evolution over time. Contradicting the popular narrative, our cross-temporal analysis reveals no consistent decline in median
review quality across venues and years. We propose alternative explanations, and outline recommendations to facilitate future empirical studies of review quality.

Contact person: [Rohan Nayak](mailto:rohan.nayak@tu-darmstadt.de)
[Ilia Kuznetsov](mailto:ilia.kuznetsov@tu-darmstadt.d)

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail if you have further questions.

## Quick Start

### Download dataset
Download the dataset in the repository root from [here](https://tudatalib.ulb.tu-darmstadt.de/items/59291e08-8477-4d48-80bb-52aa717b48f6).
#### Details of each folder
- ``full`` contains all the crawled data for the 16 considered venues. This contains crawled data from OpenReview and ARR cycles.
- ``sample`` contains the itemized and sentence split versions of the review text for a maximum of 1000 reviews sampled for each venue.
- ``precomputed_metrics`` contains the scores computed for ACT, GND and REQ for the sampled reviews in each venue.
- ``human_eval`` has all the human annotations done for validating our framework.


### Install requirements

First, ensure you have python 3.12\
Then, install the necessary requirements
```bash
uv pip install -r requirements.txt
````

For OpenAI-based itemization, set your key:

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE
```

Ensure you have access to an appropriate GPU in case you are performing inference (we used 1 A100 40GB GPU).

## Pipeline details
### Data Crawling
``data_collection`` folder contains ``openReviewCrawler.ipynb`` which contains the code used for crawling data from OpenReview, filtering and saving the data.

### Itemization + Sentence splitting
Make the following changes in ``preproc_script.sh``:
1. Activate the environment as installed.
2. Select a venue (Venue names are of types [Conference name][year], e.g., ARR2025, NeurIPS2021, ICLR2018, etc.)

Run the script with ``bash preproc_script.sh`` or with sbatch after filling the details for the SLURM configuration.

### Computing ACT and GND

Since we are reusing the model published [here](https://arxiv.org/abs/2509.04484), please make the following changes in ``act_and_gnd_script.sh``:
1. Create an environment based on the corresponding [repository](https://github.com/bodasadallah/RevUtil) and mention the environment location in the script.
2. Select a venue.

Run the script with ``bash act_and_gnd_script.sh`` or with sbatch after filling the details.

### Computing REQ
We reuse the PragTag model disccused in [this paper](https://aclanthology.org/2023.argmining-1.21/), make the following changes in ``req_script.sh``:
1. Create an environment based on this [repository]([https://github.com/bodasadallah/RevUtil](https://github.com/UKPLab/pragtag2023/tree/main)) and mention the environment location in the script.
2. Select a venue.

Run the script with ``bash req_script.sh`` or with sbatch after filling the details.

## Analysis
We already provide all the necessary files for the 16 venues we have considered in our dataset. This data is input for ``analysis.ipynb`` notebook which aggregates and computes all the 6 automatic measurements and then calculates the Q scores. The notebook contains code for all the plots used in the paper along with code used for bootstrap statistical testing. 

## Cite

Please use the following citation:

```
@misc{kuznetsov2026peerreviewreallydecline,
      title={Is Peer Review Really in Decline? Analyzing Review Quality across Venues and Time}, 
      author={Ilia Kuznetsov and Rohan Nayak and Alla Rozovskaya and Iryna Gurevych},
      year={2026},
      eprint={2601.15172},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.15172}, 
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
