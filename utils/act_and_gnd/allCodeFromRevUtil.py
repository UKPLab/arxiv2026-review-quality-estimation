import json
import re
import krippendorff
import ast
from sklearn.metrics import  f1_score,cohen_kappa_score
from scipy import stats
import numpy as np
accepted_annotators =  {
    'boda' : "boda",
    '6740484e188a64793529ee77' : "Annotator1",
    '6686ebe474531e4a1975636f': "Annotator2"
 }


# Functions used to calculate metrics. Taken from https://github.com/bodasadallah/RevUtil/blob/main/utils.py with small changes like not saving results to a file and skipping evaluation of rationales.
def write_stats_to_file(label_dict):

    results_dict = {}
    
    ########## Get stats for all items in the dict
    for key in label_dict:
        results_dict[key] = {}
        for d in label_dict[key]:

            gold = d['gold']
            preds = d['preds']
            aspect = d['aspect']
            print('Calculating the stats for', aspect)
            results_dict[key][aspect] = {}

            ### check if the gold label is dict, then we do pair-wise comparison
            is_dict = False
            try:
                if type(ast.literal_eval(gold[0])) == dict:
                    is_dict = True
            except Exception as e:
                print(e)
                print('Could not evaluate the gold label, it is not a dict')
                is_dict = False

            if is_dict:
                print('The gold label is a dict, doing pair-wise comparison')


                annotations = {}
                for i in range(len(gold)):
                    gold[i] = ast.literal_eval(gold[i])
                    for j in range(len(gold[i]['annotators'])):
                        annotator = gold[i]['annotators'][j]
                        annotator = accepted_annotators[annotator]
                        if annotator not in annotations:
                            annotations[annotator] = []
                        label = str(gold[i]['labels'][j])
                        annotations[annotator].append(label)
                
                for annotator in annotations:
                    assert len(annotations[annotator]) == len(preds), 'The number of predictions and gold labels do not match'

                    stat_dict = get_stats(preds, annotations[annotator], aspect)
                    stat_dict = process_stat_dict(stat_dict)
                    results_dict[key][aspect][annotator] = stat_dict


                # Calculate the average stat_dict across annotators
                average_stat_dict = {}
                num_annotators = len(annotations)
                for annotator in annotations:
                    for metric, value in results_dict[key][aspect][annotator].items():
                        if metric not in average_stat_dict:
                            average_stat_dict[metric] = 0.0
                        average_stat_dict[metric] += value

                for metric in average_stat_dict:
                    average_stat_dict[metric] /= num_annotators

                # Calculate Krippendorff's alpha
                annotations_plus_predictions = list(annotations.values()) + [preds]
                alpha = get_alpha_scores(annotations_plus_predictions, aspect)
                average_stat_dict['krippendorff_alpha'] = alpha

                average_stat_dict = process_stat_dict(average_stat_dict)

                results_dict[key][aspect]['total_stats'] = average_stat_dict
          
            ############## if the gold label is not a dict, we only have one label ##############
            ################### This happens for evlauation against the Test set ##################
            else:
                stat_dict = process_stat_dict(get_stats(preds, gold, aspect))

                gold_rationales = d.get('gold_data_rationale', None)
                preds_rationales = d.get('preds_rationale', None)

                rationale_results = None
                if gold_rationales:

                    # rationale_results = evaluate_rationale(gold_rationales, preds_rationales, pred_scores = preds, gold_scores = gold, aspect = aspect)
                    rationale_results = None

                # Add all keys from rationale_results to stat_dict
                if rationale_results:
                    for k, value in rationale_results.items():
                        stat_dict[k] = value


                stat_dict = process_stat_dict(stat_dict)

                results_dict[key][aspect]['total_stats'] = stat_dict

    return results_dict
    # with open(results_file_name, 'w') as f:
    #     f.write(json.dumps(results_dict, indent=4))
    #     print('Results saved to', results_file_name)


def get_alpha_scores(annotations_plus_predictions, aspect):
    possible_labels = ['1', '2', '3', '4', '5'] if aspect != 'verifiability' else ['1', '2', '3', '4', '5', 'X']
    assert len(annotations_plus_predictions) >= 2, 'There should be 4 annotators'

    # Filter out entries where any of the four labels is not in the possible labels
    filtered_annotations = []
    for aa in annotations_plus_predictions:
        if all(label in possible_labels for label in aa):
            # filtered_annotations.append([int(label) if label != "X" else 0 for label in aa])
            ## as we already consider the cases where one annotation is X in the F1 score, here we don't need to treat them as 
            filtered_annotations.append([int(label) if label != "X" else np.nan for label in aa])


    if not filtered_annotations:
        raise ValueError("No valid annotations found after filtering.")

    alpha = krippendorff.alpha(filtered_annotations, level_of_measurement='ordinal')

    return alpha


def get_stats(pred, gold, aspect):
    stats_dict = {}

    original_len = len(pred)
    ### Filter out the labels that are not in the possible labels
    possible_labels = [ '1', '2', '3', '4', '5'] if aspect != 'verifiability' else ['1', '2', '3', '4', '5', 'X']
    filtered_pred = []
    filtered_gold = []
    for i in range(len(pred)):

        if pred[i] in possible_labels and gold[i] in possible_labels:
            filtered_pred.append(pred[i])
            filtered_gold.append(gold[i])
    ### Filter out the labels that are not in the possible labels   
    pred = filtered_pred
    gold = filtered_gold
    filtered_len = len(pred)
    if aspect in ['actionability', 'grounding_specificity', 'helpfulness']:
        gold = [int(x) for x in gold]
        pred = [int(x) for x in pred]
    elif aspect == 'verifiability':
        new_pred = []
        new_gold = []
        new_pred_X = []
        new_gold_X = []
        for x, y in zip(pred, gold):
            x = str(x)
            y = str(y)
        
            if x in ['X', 'x', 'NO CLAIM']: x = 'X'
            if y in ['X', 'x', 'NO CLAIM']: y = 'X'

            if x == 'X' or y == 'X':
                x = 0 if x == 'X' else 1
                y = 0 if y == 'X' else 1
                new_pred_X.append(x)
                new_gold_X.append(y)
            else:
                x = int(x)
                y = int(y)
                new_pred.append(x)
                new_gold.append(y)
        gold = new_gold
        pred = new_pred
        stats_dict['f1_X'] = f1_score(new_pred_X, new_gold_X, average="micro")

    elif aspect in ["professional_tone", 'valid_point', 'addressed_to_author']:
        stats_dict['f1'] = f1_score(pred, gold, average="micro")



    stats_dict['kappa_quadratic'] = cohen_kappa_score(pred, gold, weights='quadratic')
    stats_dict['alpha_pairwise'] = krippendorff.alpha([pred, gold], level_of_measurement='ordinal')
    stats_dict['spearman'] = stats.spearmanr(pred, gold)
    stats_dict['pearson'] = stats.pearsonr(pred, gold)
    stats_dict['tau'] = stats.kendalltau(pred, gold)
    stats_dict['original_len'] = original_len
    stats_dict['filtered_len'] = filtered_len
    stats_dict['sucess_rate'] = filtered_len / original_len
    
    return stats_dict

def process_stat_dict(stat_dict):
    processed_stat_dict = {}
    for k, v in stat_dict.items():
        # if 'accuracy' in k:
        #     continue
        if 'spearman' in k or 'pearson' in k or 'tau' in k:
            v = v[0] if isinstance(v, tuple) else v
        if isinstance(v, float):
            v = round(v, 3)
        processed_stat_dict[k] = v

    return processed_stat_dict





# Obtained from https://github.com/bodasadallah/RevUtil/blob/main/inference/inference_utils.py
def replace_category_names(text):
    category_mapping = {
        "1: Unverifiable": 1,
        "2: Borderline Verifiable": 2,
        "3: Somewhat Verifiable": 3,
        "4: Mostly Verifiable": 4,
        "5: Fully Verifiable": 5,
        "X: No Claim": "X",

        "1: Unactionable": 1,
        "2: Borderline Actionable": 2,
        "3: Somewhat Actionable": 3,
        "4: Mostly Actionable": 4,
        "5: Highly Actionable": 5,

        "1: Not Grounded": 1,
        "2: Weakly Grounded and Not Specific": 2,
        "3: Weakly Grounded and Specific": 3,
        "4: Fully Grounded and Under-Specific": 4,
        "5: Fully Grounded and Specific": 5,

        "1: Not Helpful at All": 1,
        "2: Barely Helpful": 2,
        "3: Somewhat Helpful": 3,
        "4: Mostly Helpful": 4,
        "5: Highly Helpful": 5
    }

    # Normalize dictionary for case-insensitive matching
    category_mapping_lower = {k.lower(): v for k, v in category_mapping.items()}
    partial_mapping_lower = {k.split(": ", 1)[-1].lower(): v for k, v in category_mapping.items()}

    def replace_match(match):
        matched_text = match.group(0).lower()  # Normalize case
        return str(category_mapping_lower.get(matched_text, partial_mapping_lower.get(matched_text, match.group(0))))

    # Replace full matches first (case-insensitive)
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, category_mapping_lower.keys())) + r')\b', re.IGNORECASE)
    text = pattern.sub(replace_match, text)

    # Replace partial matches (case-insensitive)
    pattern_partial = re.compile(r'\b(?:' + '|'.join(map(re.escape, partial_mapping_lower.keys())) + r')\b', re.IGNORECASE)
    text = pattern_partial.sub(replace_match, text)

    return text

def escape_inner_quotes(text):
    """Finds specified rationale fields and escapes only internal double quotes."""
    fields = [
        "actionability_rationale",
        "grounding_specificity_rationale",
        "verifiability_rationale",
        "helpfulness_rationale"
    ]
    
    for field in fields:
        pattern = fr'("{field}"\s*:\s*")(.*?)("[\}},])'  # Escape closing brace
        matches = list(re.finditer(pattern, text, re.DOTALL))  # Find all matches first
        
        for match in reversed(matches):  # Process from last to first to avoid index shifting
            before, rationale, after = match.groups()
            escaped_rationale = rationale.replace('"', '\\"')  # Escape only inner quotes
            text = text[:match.start(2)] + escaped_rationale + text[match.end(2):]
    
    return text

def extract_dict(text):
    text = replace_category_names(text)  # Replace category names with numbers
    ## remove double spaces
    text = re.sub(' +', ' ', text)
    ## remove ``` from the text
    # text = text.replace('```', '')
    text = text.replace("-", "")  # Remove leading hyphens
    text = text.replace("\n", " ")  # Remove newlines
    text = text.replace("\\'", "'")  # Fix incorrectly escaped single quotes
    text = text.replace('\\"s', "'s")  # Fix incorrect escaped possessive 's
    text = text.replace("\\\\'", "\\\"")
    text = text.replace("\\\\", "\\")

    text = text.replace("[", "")  # Remove square brackets
    text = text.replace("]", "")  # Remove square brackets

    ############## For Prometheus2 #################
    # text = text.replace("[", '"')  # Replace single quotes with double quotes
    # text = text.replace("]", '"')  # Replace single quotes with double quotes

    ## if text begin with comma or space, remove it
    if text[0] == ',' or text[0] == ' ':
        text = text[1:]

    text = text.replace("\\\\", "\\") # Fix double backslashes
    dict_str  = "" 
    if "```" in text:
        text = text + '#'
        match = re.search(r"```(?:json)?\s*(.*?)(```)?#", text, re.DOTALL)
        if match:
            text = match.group(0)
            ## remove the ```json  and ``` from the text
            text = text.replace('```json', '')
            text = text.replace('```', '')
            text = text.replace('#', '') 

    text = text.strip()  # Remove leading and trailing whitespace

    if not text:
        return None
    
    ############# Comment for Prometheus2 ##############
    if text[0] != '{':
        text = '{' + text + '}'

    ################## cut the text if there is two newlines. This is for Prmetheus2 #########
    # if '\n\n' in text:
    #     halfs = text = text.split('\n\n', 1)
    #     text = halfs[0] if "actionability_label" in halfs[0] else halfs[1]

    if '{' not in text:
        text = '{' + text + '}'

    text = text.replace(" }  { ", ',')  # Remove newlines between dictionaries

    text = text.replace("\n", ' ')  # Remove newlines

    ############################# Prometheus2 ##########################
    # text = extract_valid_json(text)
    # text = text.replace('\\', '')  # Remove single quotes
    # print(f"Text after processing: {text} \n\n\n")

    ############ Some cases doesn't work with replacing the quotes, so trying both ways
    text2 = text
    try:
        text = text.replace("'", '"')  # Replace single quotes with double quotes
        text = escape_inner_quotes(text)  # Fix quotes inside rationale fields
        match = re.search(r'\{.*?\}', text, re.DOTALL)  # Extract first {...} block
        if match:
            dict_str = match.group()  # Get extracted dictionary string
        return json.loads(dict_str)  # Convert to Python dictionary safely
    except json.JSONDecodeError:
        print("Replacing quotes didn't work, trying without replacing quotes.")
        text = text2  # Revert to original text
        match = re.search(r'\{.*?\}', text, re.DOTALL)  # Extract first {...} block
        if match:
            dict_str = match.group()  # Get extracted dictionary string
        try:
            return json.loads(dict_str)  # Convert to Python dictionary safely
        except json.JSONDecodeError as e:
            print(f"Parsing error: {e}\nProblematic string: {dict_str}")
            return None

def extract_predictions(model_outputs):
    """
    Parses a list of model-generated texts to extract labels and returns a dictionary.
    
    :param model_outputs: List of strings containing model-generated text with labels.
    :return: List of dictionaries with extracted labels.
    """
    extracted_data = []

    for model_output in model_outputs:
        text = model_output.outputs[0].text

        # if 'outputs' in text.keys():
        #     text = text.outputs[0].text
        # elif 'generated_text' in text.keys():
        #     text = text['generated_text']

        extracted_dict = extract_dict(text)
        if  not extracted_dict:
            extracted_data.append({
                'actionability_label': None,
                'grounding_specificity_label': None,
                'verifiability_label': None,
                'helpfulness_label': None,
                'actionability_rationale': None,
                'grounding_specificity_rationale': None,
                'verifiability_rationale': None,
                'helpfulness_rationale': None
            })
            continue

        parsed_result = {
            'actionability_label': str(extracted_dict.get('actionability_label', None)),
            'grounding_specificity_label':  str(extracted_dict.get('grounding_specificity_label', None)),
            'verifiability_label':  str(extracted_dict.get('verifiability_label', None)),
            'helpfulness_label':  str(extracted_dict.get('helpfulness_label', None)),
            ### rationale keys
            'actionability_rationale':  str(extracted_dict.get('actionability_rationale', None)),
            'grounding_specificity_rationale':  str(extracted_dict.get('grounding_specificity_rationale', None)),
            'verifiability_rationale':  str(extracted_dict.get('verifiability_rationale', None)),
            'helpfulness_rationale':  str(extracted_dict.get('helpfulness_rationale', None))
        }

        extracted_data.append(parsed_result)
    
    return extracted_data





# Adapted based on https://github.com/bodasadallah/RevUtil/blob/main/utils.py#L49-L217
def get_prompt(review_point):
    prompt_header = PROMPT_HEADER

    aspects = [ 'actionability', 'grounding_specificity', 'verifiability', 'helpfulness']
    considered_aspects = aspects
    aspect_definitions = ''
    for aspect in considered_aspects:
        aspect_definition = ASPECTS_NO_EXAMPLES[aspect]
        aspect_definitions += f'''Aspect: {aspect}\n{aspect_definition}\n'''

    prompt = f'''###Task Description:
{prompt_header}

{aspect_definitions}
'''
    
    prompt += INSTRUCTION_SCORE_ONLY_PROMPT_TAIL.format(review_point=review_point)

    prompt += '''\n\n###Output:\n'''
    return prompt



PROMPT_HEADER = '''You are an expert in evaluating peer review comments with respect to different aspects. These aspects are aimed to maximize the utilization of the review comments for the authors. The primary purpose of the review is to help/guide authors in improving their drafts. Keep this in mind while evaluating the review point. Whenever you encounter a borderline case, think: “Will this review point help authors improve their draft?”. There is no correlation between the aspect score and the length of the review point.'''


ASPECTS_NO_EXAMPLES = {
"actionability": 
'''
**Actionability**

**Definition:** Measures the level of actionability in a review point. We evaluate actionability based on two criteria:

1. **Explicit vs. Implicit**:
   - **Explicit:** Actions or suggestions that are direct or apparent. Authors can directly identify modifications they should apply to their draft. Clarification questions should be treated as explicit statements if they give a direct action.
   - **Implicit:** Actions that need to be inferred from the comment. This includes missing parts that need to be added. Authors can deduce what needs to be done after reading the comment.

2. **Concrete vs. Vague**:
   - **Concrete:** Once the action is identified, the authors know exactly what needs to be done and how to apply the action.
   - **Vague:** After identifying the action, the authors still don’t know how to carry out this action.

**Importance:** It’s more important for actions to be concrete so that authors know how to apply them. It's also preferred for actions to be stated directly rather than inferred.

**Actionability Scale (1-5):**

1. **1: Unactionable**
   - **Definition:** The comment lacks meaningful information to help authors improve the paper. Authors do not know what they should do after reading the comment.

2. **2: Borderline Actionable**
   - **Definition:** The comment includes an implicitly stated action or an action that can be inferred. However, the action itself is vague and lacks detail on how to apply it.

3. **3: Somewhat Actionable**
   - **Definition:** The comment explicitly states an action but is vague on how to execute it.

4. **4: Mostly Actionable**
   - **Definition:** The comment implicitly states an action but concretely states how to implement the inferred action.

5. **5: Highly Actionable**
   - **Definition:** The comment contains an explicit action and concrete details on how to implement it. Authors know exactly how to apply it.
''',


"grounding_specificity": '''

**Grounding Specificity**  

**Definition:** Measures how explicitly a review comment refers to a specific part of the paper and how clearly it identifies the issue with that part. This helps authors understand what needs revision and why. Grounding specificity has two key components:  

1. **Grounding:** How well the authors can identify the specific part of the paper being addressed.  
   - **Weak Grounding:** The author can make an educated guess but cannot precisely identify the referenced part.  
   - **Full Grounding:** The author can accurately pinpoint the section, table, figure, or unique aspect being addressed. This can be achieved through:  
     - Literal mentions of sections, tables, figures, etc.  
     - Mentions of unique elements of the paper.  
     - General comments that clearly imply the relevant parts without explicitly naming them.  

2. **Specificity:** How clearly the comment details what is wrong or missing in the referenced part. If external work is mentioned, it also evaluates whether specific examples are provided.  

**Importance:** It's more important for the comment to be grounded than to be specific.  

**Grounding Specificity Scale (1-5):** 

1. **Not Grounded**
   - **Definition**: This comment is not grounded at all. It does not identify a specific area in the paper. The comment is highly unspecific.

2. **Weakly Grounded and Not Specific**
   - **Definition**: The authors cannot confidently determine which part the comment addresses. Further, the comment does not specify what needs to be addressed in this part.

3. **Weakly Grounded and Specific**
   - **Definition**: The authors cannot confidently determine which part the comment addresses. However, the comment clearly specifies what needs to be addressed in this part.

4. **Fully Grounded and Under-Specific**
   - **Definition**: The comment explicitly mentions which part of the paper it addresses, or it should be obvious to the authors. However, this comment does not specify what needs to be addressed in this part.

5. **Fully Grounded and Specific**
   - **Definition**: The comment explicitly mentions which part of the paper it addresses, and it is obvious to the authors. The comment specifies what needs to be addressed in this part.
''',
    
"verifiability":  
'''  
**Verifiability**  

**Definition:** Evaluates whether a review comment contains a claim and, if so, how well that claim is supported using logical reasoning, common knowledge, or external references.  

### **Step 1: Claim Extraction**  

**Objective:**  
Determine whether the given text contains a claim (i.e., an opinion, judgment, or suggestion) or consists solely of factual statements that require no verification.  

**Claim Definition:**  
A statement is considered a claim if it falls into one or more of the following categories:  
- **Subjective opinions or disagreements** (e.g., criticism of an experimental choice).  
- **Suggestions or requests for changes** (e.g., recommending removal, addition, or discussion).  
- **Judgments about the paper** (e.g., stating something is unclear, not well-written, or lacks detail).  
- **Deductions or inferred observations** that go beyond merely stating facts.  
- **Statements requiring justification** to be understood or accepted.  


**Normal Statements ("No Claim")**  
A statement is classified as "X" if it:  
- Describes facts without suggesting changes.  
- Makes general statements about the paper without an opinion.  
- Presents objective, verifiable facts that require no justification.  
- Asks for clarifications or general questions.  
- States logical statements or directly inferable information.  
- Makes positive claims (e.g., *“The paper is well-written”*), as these do not help improve the work.  

---  

### **Step 2: Verifiability Verification**  

**Objective:**  
Assess how well a claim is verified by examining the reasoning, common knowledge, or external references provided. The purpose is to ensure that the review comment helps the authors improve their work.  

**Verification Methods:**  
A claim is considered verifiable if supported by one or more of the following:  
- **Logical reasoning** – A clear explanation of why the claim is valid.  
- **Common knowledge** – Reference to well-accepted practices or standards.  
- **External references** – Citation of relevant literature, data, or sources.  

**Verifiability Scale (1–5 & X):**  

1. **1: Unverifiable**  
   - The comment contains a claim without any supporting evidence or justification.  
2. **2: Borderline Verifiable**  
   - Some support is provided, but it is vague, insufficient, or difficult to follow.  
3. **3: Somewhat Verifiable**  
   - The claim has some justification but lacks key elements (e.g., examples, references).  
4. **4: Mostly Verifiable**  
   - The claim is well-supported but has minor gaps in explanation or references.  
5. **5: Fully Verifiable**  
   - The claim is thoroughly supported by explicit, sufficient, and robust evidence, such as:  
     - Clear reasoning and precise explanations.  
     - Specific references to external works.  
     - Logical and unassailable common-sense arguments.  
6. **X: No Claim**  
- The comment contains only factual, descriptive statements without claims, opinions, or suggestions.  

---  

### **Instructions for Evaluation:**  
1. **Extract Claims:** Determine whether the review comment contains a claim or is a normal statement. If there is no claim, score it as "X"  
2. **Assess Verifiability:** If a claim exists, score it based on how well it is justified from 1 to 5.  
''',

"helpfulness" : '''
**Helpfulness**

**Definition:** Assign a subjective score to reflect the value of the review comment to the authors. Helpfulness is rated on a scale from 1 to 5, with the following definitions:

1. **1: Not Helpful at All**  
   - **Definition:** The comment fails to identify meaningful weaknesses or suggest improvements, leaving the authors with no actionable feedback.  

2. **2: Barely Helpful**  
   - **Definition:** The comment identifies a weakness or improvement area but is vague, lacks clarity, or provides minimal guidance, making it only slightly beneficial for the authors.  

3. **3: Somewhat Helpful**  
   - **Definition:** The comment identifies weaknesses or areas for improvement but is incomplete or lacks depth. While the authors gain some insights, the feedback does not fully address their needs for improving the draft.  

4. **4: Mostly Helpful**  
   - **Definition:** The comment provides clear and actionable feedback on weaknesses and areas for improvement, though it could be expanded or refined to be fully comprehensive and impactful.  

5. **5: Highly Helpful**  
   - **Definition:** The comment thoroughly identifies weaknesses and offers detailed, actionable, and constructive suggestions that empower the authors to significantly improve their draft.  
'''

}

INSTRUCTION_SCORE_ONLY_PROMPT_TAIL = '''
###Instruction:
Evaluate the review based on the given definitions of the aspect(s) above. Output only the score.

###Review Point:
{review_point}'''