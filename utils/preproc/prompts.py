class StructurerPrompts:
    def __init__(self):
        self.examples = """# Examples
## Example Review 1 Input
Summary:
The authors introduce Conditional Activation Steering (CAST) and condition vectors. They show that refusal behaviors can be invoked conditionally on the context of the prompt allowing for conditional steering. They test this across several language models up to size 8B and show that their method has fewer false positive refusals on harmless prompts while still maintaining high refusal rates on harmful prompts, demonstrating the effectiveness of the conditional.

Strengths:
1. The method of steering LLM conditionally on the context of the prompt is novel and an important contribution towards practical implementations of activation steering.
2. The ability to chain conditionals is an interesting contribution.
3. The paper is relatively thorough in its test of models within a certain class O(8B).

Weaknesses:
1. All the tested models have less than or equal to 8B parameters. Testing on larger models would help improve the robustness and confidence in the results
2. (Minor) The harmless/harmful refusals are not tested against enough real-world inputs, like jailbreaks or multi-turn conversations.
3. (Minor) There is no limitations or future work section.

Questions:
The paper is generally well written and was pleasant and interesting to read. The possibility for conditional steering is exciting with many practical implications.

### Minor
The paper could be improved through some careful revisions to the figures and layouts.
* Figure 1 is presented too early; its full description is on page 6 while it appears at the top of page 2. Despite being referenced on page 1, the paper would flow better if Figure 1 were closer to page 6.
* All of the T-SNE plots should consider a different color scheme. Its very difficult to distinguish the Alpaca vs Sorry-bench dots, especially against the background of a similar color.
* Figure 8c, several pieces of text are too small to easily read
* Figure 9, the label "(c)" is not placed in the top left corner like the previous figures. The markers are difficult to see (e.g. the start marker).


### Out-of-scope improvements
While the following improvements would substantially increase the value of the paper, the reviewer recognizes that they can be designated to follow up work and may be out-of-scope for the current paper.
* The methods could be tested with models with > 8B parameters
* The methods could be tested against known jailbreaks (e.g. does the conditional vector for harmfulness still activation on a zero-shot or 1-shot jailbreak?).

## Example Review 1 Output
{
  "summary_segments": [
    "The authors introduce Conditional Activation Steering (CAST) and condition vectors. They show that refusal behaviors can be invoked conditionally on the context of the prompt allowing for conditional steering. They test this across several language models up to size 8B and show that their method has fewer false positive refusals on harmless prompts while still maintaining high refusal rates on harmful prompts, demonstrating the effectiveness of the conditional."
  ],
  "strengths_segments": [
    "The method of steering LLM conditionally on the context of the prompt is novel and an important contribution towards practical implementations of activation steering.",
    "The ability to chain conditionals is an interesting contribution.",
    "The paper is relatively thorough in its test of models within a certain class O(8B).",
    "The paper is generally well written and was pleasant and interesting to read. The possibility for conditional steering is exciting with many practical implications."
  ],
  "weaknesses_suggestions_questions_comments_segments": [
    "All the tested models have less than or equal to 8B parameters. Testing on larger models would help improve the robustness and confidence in the results",
    "(Minor) The harmless/harmful refusals are not tested against enough real-world inputs, like jailbreaks or multi-turn conversations.",
    "(Minor) There is no limitations or future work section.",
    "The paper could be improved through some careful revisions to the figures and layouts.\n* Figure 1 is presented too early; its full description is on page 6 while it appears at the top of page 2. Despite being referenced on page 1, the paper would flow better if Figure 1 were closer to page 6.",
    "The paper could be improved through some careful revisions to the figures and layouts.\n* All of the T-SNE plots should consider a different color scheme. Its very difficult to distinguish the Alpaca vs Sorry-bench dots, especially against the background of a similar color.",
    "The paper could be improved through some careful revisions to the figures and layouts.\n* Figure 8c, several pieces of text are too small to easily read",
    "The paper could be improved through some careful revisions to the figures and layouts.\n* Figure 9, the label "(c)" is not placed in the top left corner like the previous figures. The markers are difficult to see (e.g. the start marker).",
    "While the following improvements would substantially increase the value of the paper, the reviewer recognizes that they can be designated to follow up work and may be out-of-scope for the current paper.\n* The methods could be tested with models with > 8B parameters",
    "While the following improvements would substantially increase the value of the paper, the reviewer recognizes that they can be designated to follow up work and may be out-of-scope for the current paper.\n* The methods could be tested against known jailbreaks (e.g. does the conditional vector for harmfulness still activation on a zero-shot or 1-shot jailbreak?)."
  ],
  "other_segments": []
}


## Example Review 2 Input
Summary:
This paper explores interleaved multimodal conversations with LLMs. The authors first construct a multimodal conversation instruction tuning dataset with text-only GPT-4 and image descriptions. Then, multi-turn interleaved multimodal (dubbed MIM) instruction tuning based on GILL is used to train the multimodal LLMs to learn which segment of text information should be used for diffusion model image synthesis (which also defines when to generate images). Experiments demonstrate promising results.

Strengths:
- The targeted problem of enabling multimodal LLMs to generate images for a multimodal multi-turn conversation is trending, interesting, and important. Great application potentials can be induced in both the research community and the industry.
- The proposed dataset would be very useful to the community if it is open-sourced. The construction involves human-in-the-loop refinement, which is good since the data from the internet is extremely noisy.

Weaknesses:
- **Major concern 1.** My first major concern lies in the novelty of the proposed textual exchange method for enabling LLMs to generate images. As far as I know, learning which segment of texts to use as the text inputs of text-to-image models was first proposed by Divter [1] (not cited or discussed). Divter proposes to learn the textural inputs by using a special token and constructed template. However, leveraging LLMs and diffusion models is different, but the contribution is still limited in this case. The authors are required to clarify the similarities and differences with Divter. On the other hand, using only textual conditions to generate images has its limitations when considering very long-context-based (e.g., interleaved documents) image generation or image-conditioned image generation (e.g., image2image translation, image edition, etc).
- **Major concern 2.** My second major concern lies in the experimental evaluations. i) In Table 3, more commonly used NLP benchmarks like MMLU, HellaSwag, and WinoGrande should be conducted. In Table 5, what about the most commonly used metric FID results? In Table 6, I wonder about the zero-shot results of TextBind on VQAv2 and MM-Vet. ii) No ablation studies or in-depth discussions are presented. For example, what if no human-in-the-loop is used during dataset construction? What are the failure cases of TextBind, and why? What emerging properties could be explored by TextBind?
- **Technical contribution.** The proposed method is simple, but the technical contribution is limited. The proposed multimodal LLM architecture is mainly based on previous work GILL. The only difference is the textural information exchange method which is good but somewhat incremental. Besides, the topic-aware image sampling is quite similar to the dataset constructed by PALI-X [2]. PALI-X constructs its own interleaved dataset Episodic WebLI by grouping image-text pairs.
- **Unclearly supported claims.** The authors claim that the current multimodal instruction tuning methods lead to limited performance in open-world scenarios. However, there is a lack of analysis of TextBind's superiority in it. Besides, annotation-free may be overclaimed since human-in-the-loop definitely requires non-trivial annotation efforts.

[1] Multimodal Dialogue Response Generation. In ACL 2022.
[2] PaLI-X: On Scaling up a Multilingual Vision and Language Model.

Questions:
- Why is the method called TextBind? Assuming the authors are trying to analog to ImageBind [3]. However, the core spirit of multimodal binding in the same embedding space as a modality-agnostic multimodal encoder is very different from this paper. I am quite confused about this. A name that better summarizes the work's idea is better than using one similar to an existing work while not very suitable.
- Will the curated dataset be released to the public?
- The dataset is constructed by using image descriptions with GPT-4, similar to LLaVA [4] and ChatCaptioner [5] (not cited). I wonder how is the hallucination problem of data and models in such progress since CLIP filtering can not guarantee the avoidance of such an issue. For example, can authors provide some failure cases of the dataset and test the model's hallucination capability?
- Now we have GPT-4V, I am looking forward to the newly constructed dataset with GPT-4V (not required at this moment).
- Since Q-Former is used, which may compress the visual signals, I wonder about the OCR performance of TextBind. For example, zero-shot results on TextVQA?
- There are many concurrent works working in this direction. It would be good if these works were discussed in related work [6-10].
- Minor: When the abbreviation `MIM` first appears in the paper, there is no explanation of the meanings.

I am looking forward to the authors' response.

[3] ImageBind: One Embedding Space To Bind Them All. In CVPR 2023.
[4] Visual instruction tuning. In NeurIPS 2023.
[5] ChatGPT Asks, BLIP-2 Answers: Automatic Questioning Towards Enriched Visual Descriptions. arXiv 2023.
[6] Generative Pretraining in Multimodality. arXiv 2023.
[7] DreamLLM: Synergistic Multimodal Comprehension and Creation. arXiv 2023.
[8] MiniGPT-5: Interleaved Vision-and-Language Generation via Generative Vokens. arXiv 2023.
[9] NExT-GPT: Any-to-Any Multimodal LLM. arXiv 2023.
[10] Kosmos-G: Generating Images in Context with Multimodal Large Language Models. arXiv 2023.

## Example Review 2 Output
{
  "summary_segments": [
    "This paper explores interleaved multimodal conversations with LLMs. The authors first construct a multimodal conversation instruction tuning dataset with text-only GPT-4 and image descriptions. Then, multi-turn interleaved multimodal (dubbed MIM) instruction tuning based on GILL is used to train the multimodal LLMs to learn which segment of text information should be used for diffusion model image synthesis (which also defines when to generate images). Experiments demonstrate promising results."
  ],
  "strengths_segments": [
    "The targeted problem of enabling multimodal LLMs to generate images for a multimodal multi-turn conversation is trending, interesting, and important. Great application potentials can be induced in both the research community and the industry.",
    "The proposed dataset would be very useful to the community if it is open-sourced. The construction involves human-in-the-loop refinement, which is good since the data from the internet is extremely noisy."
  ],
  "weaknesses_suggestions_questions_comments_segments": [
    "**Major concern 1.** My first major concern lies in the novelty of the proposed textual exchange method for enabling LLMs to generate images. As far as I know, learning which segment of texts to use as the text inputs of text-to-image models was first proposed by Divter [1] (not cited or discussed). Divter proposes to learn the textural inputs by using a special token and constructed template. However, leveraging LLMs and diffusion models is different, but the contribution is still limited in this case. The authors are required to clarify the similarities and differences with Divter. On the other hand, using only textual conditions to generate images has its limitations when considering very long-context-based (e.g., interleaved documents) image generation or image-conditioned image generation (e.g., image2image translation, image edition, etc).",
    "**Major concern 2.** My second major concern lies in the experimental evaluations. i) In Table 3, more commonly used NLP benchmarks like MMLU, HellaSwag, and WinoGrande should be conducted. In Table 5, what about the most commonly used metric FID results? In Table 6, I wonder about the zero-shot results of TextBind on VQAv2 and MM-Vet. ii) No ablation studies or in-depth discussions are presented. For example, what if no human-in-the-loop is used during dataset construction? What are the failure cases of TextBind, and why? What emerging properties could be explored by TextBind?",
    "**Technical contribution.** The proposed method is simple, but the technical contribution is limited. The proposed multimodal LLM architecture is mainly based on previous work GILL. The only difference is the textural information exchange method which is good but somewhat incremental. Besides, the topic-aware image sampling is quite similar to the dataset constructed by PALI-X [2]. PALI-X constructs its own interleaved dataset Episodic WebLI by grouping image-text pairs.",
    "**Unclearly supported claims.** The authors claim that the current multimodal instruction tuning methods lead to limited performance in open-world scenarios. However, there is a lack of analysis of TextBind's superiority in it. Besides, annotation-free may be overclaimed since human-in-the-loop definitely requires non-trivial annotation efforts.",
    "Why is the method called TextBind? Assuming the authors are trying to analog to ImageBind [3]. However, the core spirit of multimodal binding in the same embedding space as a modality-agnostic multimodal encoder is very different from this paper. I am quite confused about this. A name that better summarizes the work's idea is better than using one similar to an existing work while not very suitable.",
    "Will the curated dataset be released to the public?",
    "The dataset is constructed by using image descriptions with GPT-4, similar to LLaVA [4] and ChatCaptioner [5] (not cited). I wonder how is the hallucination problem of data and models in such progress since CLIP filtering can not guarantee the avoidance of such an issue. For example, can authors provide some failure cases of the dataset and test the model's hallucination capability?",
    "Now we have GPT-4V, I am looking forward to the newly constructed dataset with GPT-4V (not required at this moment).",
    "Since Q-Former is used, which may compress the visual signals, I wonder about the OCR performance of TextBind. For example, zero-shot results on TextVQA?",
    "There are many concurrent works working in this direction. It would be good if these works were discussed in related work [6-10].",
    "Minor: When the abbreviation `MIM` first appears in the paper, there is no explanation of the meanings."
  ],
  "other_segments": [
    "[1] Multimodal Dialogue Response Generation. In ACL 2022.\n[2] PaLI-X: On Scaling up a Multilingual Vision and Language Model.",
    "I am looking forward to the authors' response.",
    "[3] ImageBind: One Embedding Space To Bind Them All. In CVPR 2023.\n[4] Visual instruction tuning. In NeurIPS 2023.\n[5] ChatGPT Asks, BLIP-2 Answers: Automatic Questioning Towards Enriched Visual Descriptions. arXiv 2023.\n[6] Generative Pretraining in Multimodality. arXiv 2023.\n[7] DreamLLM: Synergistic Multimodal Comprehension and Creation. arXiv 2023.\n[8] MiniGPT-5: Interleaved Vision-and-Language Generation via Generative Vokens. arXiv 2023.\n[9] NExT-GPT: Any-to-Any Multimodal LLM. arXiv 2023.\n[10] Kosmos-G: Generating Images in Context with Multimodal Large Language Models. arXiv 2023."
  ]
}


## Example Review 3 Input
Review:
The paper proposed a RNN with skip-connection (external memory) to past hidden states, this is a slightly different version of the TARDIS network. The authors experimented on PTB and a temporal action detection method.

Novelty:

I dont see a lot of novelty to the method. The authors proposed a method very similar to TARDIS, the difference seems to be that MMARNN does not use extra usage vectors for reading from previous memory, but this is not a fundamental difference between MMARNN and Tardis.

Shortcomings of the paper:

1. The experiments seem rather weak. The authors experimented on PTB and temporal action detection method. It is not clear why authors experimented with PTB, this is not a task with long-term dependencies, I do not see how this task (compared to many other tasks) can benefit from using external memory (especially when only 1 past hidden state is used

2. The model uses a single past hidden state, it is not clear to me why this is better than using a weighted sum of a few past hidden states, as many tasks requires long-term dependencies from multiple steps in the past. The authors should cite "Sparse attentive backtracking" (https://arxiv.org/abs/1809.03702) at NIPS 2018. SAB is very related in that it also propagate gradients to a few hidden states in the memory. The difference is that SAB used a few hidden states from the past/ memory instead of one; another difference is that it propagates gradients locally to the selected hidden states/ memory slots.

3. The paper only demonstrated experimental results on PTB and temporal action prediction. I think it would make the paper a lot stronger if the authors experimented with a variety of different tasks. Tasks that requires long term dependencies can really demonstrate the strength of the model (copy and adding tasks).

4. If the authors could run the model on copy and adding tasks, I would be curious to see if the model is picking the "correct" timestep in the memory / past.

post rebuttal: I feel that the authors have addressed some of my concerns, in particular, in terms of additional experimental results. I have raised the score to reflect this changes.

## Example Review 3 Output
{
  "summary_segments": [
    "The paper proposed a RNN with skip-connection (external memory) to past hidden states, this is a slightly different version of the TARDIS network. The authors experimented on PTB and a temporal action detection method."
  ],
  "strengths_segments": [],
  "weaknesses_suggestions_questions_comments_segments": [
    "I dont see a lot of novelty to the method. The authors proposed a method very similar to TARDIS, the difference seems to be that MMARNN does not use extra usage vectors for reading from previous memory, but this is not a fundamental difference between MMARNN and Tardis.",
    "The experiments seem rather weak. The authors experimented on PTB and temporal action detection method. It is not clear why authors experimented with PTB, this is not a task with long-term dependencies, I do not see how this task (compared to many other tasks) can benefit from using external memory (especially when only 1 past hidden state is used",
    "The model uses a single past hidden state, it is not clear to me why this is better than using a weighted sum of a few past hidden states, as many tasks requires long-term dependencies from multiple steps in the past. The authors should cite \"Sparse attentive backtracking\" (https://arxiv.org/abs/1809.03702) at NIPS 2018. SAB is very related in that it also propagate gradients to a few hidden states in the memory. The difference is that SAB used a few hidden states from the past/ memory instead of one; another difference is that it propagates gradients locally to the selected hidden states/ memory slots.",
    "The paper only demonstrated experimental results on PTB and temporal action prediction. I think it would make the paper a lot stronger if the authors experimented with a variety of different tasks. Tasks that requires long term dependencies can really demonstrate the strength of the model (copy and adding tasks).",
    "If the authors could run the model on copy and adding tasks, I would be curious to see if the model is picking the \"correct\" timestep in the memory / past."
  ],
  "other_segments": [
    "post rebuttal: I feel that the authors have addressed some of my concerns, in particular, in terms of additional experimental results. I have raised the score to reflect this changes."
  ]
}


## Example Review 4 Input
Summary:
This paper investigates the relationship between the batch-size and convergence speed for a broader class of nonconvex problems. Unfortunately, the results are not novel, the introduction is very short it isn not clear the main contribution of the paper, and the comparison with related work is quite brief without much in-depth discussion.

Strengths:
N/A

Weaknesses:
The submission is unfortunately not very strong. The main results (Theorem 1.1) shows that the effective noise level reduces with batch size, which seems trivial. Indeed, I cannot see much novelty and intuition provided by Theorem 1.1.

---------------------

The introduction is very short it isn not clear the main contribution of the paper, and the comparison with related work is quite brief without much in-depth discussion.

---------------------

The paper considers a narrow optimization problem, and the setting in Theorem 1.1 is also quite limited.

Therefore, I do recommend rejection.

Questions:
Please see above.

## Example Review 4 Output
{
  "summary_segments": [
    "This paper investigates the relationship between the batch-size and convergence speed for a broader class of nonconvex problems. Unfortunately, the results are not novel, the introduction is very short it isn not clear the main contribution of the paper, and the comparison with related work is quite brief without much in-depth discussion."
  ],
  "strengths_segments": [],
  "weaknesses_suggestions_questions_comments_segments": [
    "The submission is unfortunately not very strong. The main results (Theorem 1.1) shows that the effective noise level reduces with batch size, which seems trivial. Indeed, I cannot see much novelty and intuition provided by Theorem 1.1.",
    "The introduction is very short it isn not clear the main contribution of the paper, and the comparison with related work is quite brief without much in-depth discussion.",
    "The paper considers a narrow optimization problem, and the setting in Theorem 1.1 is also quite limited."
  ],
  "other_segments": [
    "Therefore, I do recommend rejection.",
    "Please see above."
  ]
}"""
        self.promptDict = {
            "system": "You are an expert in peer-review for scientific conferences.",
            "user": """# Task Description
You are given a review from a conference which may have a structure of its own or be completely unstructured. Your objective is to arrange the review segments into 4 sections to obtain the review in a structured format. The 4 sections are: Summary, Strengths, Weaknesses/Suggestions/Questions/Comments and Other. Strengths and Weaknesses/Suggestions/Questions/Comments need to have separated points while Summary does not. Each point should be focused on one claim and have all the context mentioned in the review for it. You may have to combine different segments which are not consecutively placed in the original review into one point to achieve this. On the other hand, it is also possible that a single text discussing multiple ideas may need to be split into different points. When doing this, make sure to repeat any text necessary so that the context is not lost. Any additional information which does not fit into Summary, Strengths and Weaknesses/Suggestions/Questions/Comments sections should be placed in the Other section. You are provided with 4 examples on how to do this task. Follow the examples closely while structuring the review.

# Rules
1. Avoid paraphrasing unless absolutely needed for coherence and do not change the meaning of any part of the review. 
2. Make sure to minimize adding new content or discarding any content from the given review. 
3. If details for a section is missing in the original review, you can leave that section empty in the output.
4. Do not correct any grammatical or spelling mistakes in the review.
5. The final output should in JSON format containing a list of strings for each section, with each string for Strengths and Weaknesses/Suggestions/Questions/Comments sections corresponding to one point.

{examples}

# Review to be structured
{review_text}"""
        }