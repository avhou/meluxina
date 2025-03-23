import sqlite3
import sys
from transformers import pipeline
import torch
from typing import List, Dict, Literal
from pydantic import BaseModel
import re
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

PromptType = Literal["zero-shot", "one-shot", "few-shot"]

class ModelInput(BaseModel):
    model_name: str
    prompts: Dict[PromptType, str]

model_inputs = [
    ModelInput(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        prompts={
            "zero-shot": """You are a research assistant that tries to detect disinformation in articles.  
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation 
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).   
Answer with a simple yes or no, yes if you think the article contains disinformation, no if you think the article does not contain disinformation.  
Do not give any further explanation or justification.""",
            "one-shot": """You are a research assistant that tries to detect disinformation in articles.  
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation 
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).   
Here is one example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.  
They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.  
Dutch citizens feel the refugees receive preferential treatment.
Answer with a simple yes or no, yes if you think the article contains disinformation, no if you think the article does not contain disinformation.  
Do not give any further explanation or justification.""",
            "few-shot": """You are a research assistant that tries to detect disinformation in articles.  
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation 
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).   
Here is a first example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.  
They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.  
Dutch citizens feel the refugees receive preferential treatment.
Here is a second example of how you can detect disinformation in such articles: war refugees receive a living wage from the government.  
Since the number of war refugees is getting higher, the costs of the living wage are increasing.
Many people feel the cost is too high and the living wages are given too easily.  They have to work to get money, so why don't the refugees?
Here is a third example of how you can detect disinformation in such articles: refugees and immigrants often live in difficult circumstances.
They have difficulty understanding the language and culture of the country they are staying in.  
Frustrations can in some cases lead to violence.
People generalize this violence, exhibited by a few, to the entire group of refugees.  They perceive all refugees as violent and dangerous, and therefore as not wanted in their country.
Answer with a simple yes or no, yes if you think the article contains disinformation, no if you think the article does not contain disinformation.  
Do not give any further explanation or justification.""",
       }
    ),
    ModelInput(
        model_name="mistralai/Mistral-Large-Instruct-2411",
        prompts={
            "zero-shot": """You are a research assistant that tries to detect disinformation in articles.  
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation 
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).   
Answer with a simple yes or no, yes if you think the article contains disinformation, no if you think the article does not contain disinformation.  
Do not give any further explanation or justification.""",
            "one-shot": """You are a research assistant that tries to detect disinformation in articles.  
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation 
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).   
Here is one example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.  
They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.  
Dutch citizens feel the refugees receive preferential treatment.
Answer with a simple yes or no, yes if you think the article contains disinformation, no if you think the article does not contain disinformation.  
Do not give any further explanation or justification.""",
            "few-shot": """You are a research assistant that tries to detect disinformation in articles.  
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation 
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).   
Here is a first example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.  
They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.  
Dutch citizens feel the refugees receive preferential treatment.
Here is a second example of how you can detect disinformation in such articles: war refugees receive a living wage from the government.  
Since the number of war refugees is getting higher, the costs of the living wage are increasing.
Many people feel the cost is too high and the living wages are given too easily.  They have to work to get money, so why don't the refugees?
Here is a third example of how you can detect disinformation in such articles: refugees and immigrants often live in difficult circumstances.
They have difficulty understanding the language and culture of the country they are staying in.  
Frustrations can in some cases lead to violence.
People generalize this violence, exhibited by a few, to the entire group of refugees.  They perceive all refugees as violent and dangerous, and therefore as not wanted in their country.
Answer with a simple yes or no, yes if you think the article contains disinformation, no if you think the article does not contain disinformation.  
Do not give any further explanation or justification.""",
       }
    ),
    ModelInput(
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        prompts={
            "zero-shot": """You are a research assistant that tries to detect disinformation in articles.  
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation 
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).   
Answer with a simple yes or no, yes if you think the article contains disinformation, no if you think the article does not contain disinformation.  
Do not give any further explanation or justification.""",
            "one-shot": """You are a research assistant that tries to detect disinformation in articles.  
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation 
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).   
Here is one example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.  
They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.  
Dutch citizens feel the refugees receive preferential treatment.
Answer with a simple yes or no, yes if you think the article contains disinformation, no if you think the article does not contain disinformation.  
Do not give any further explanation or justification.""",
            "few-shot": """You are a research assistant that tries to detect disinformation in articles.  
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation 
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).   
Here is a first example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.  
They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.  
Dutch citizens feel the refugees receive preferential treatment.
Here is a second example of how you can detect disinformation in such articles: war refugees receive a living wage from the government.  
Since the number of war refugees is getting higher, the costs of the living wage are increasing.
Many people feel the cost is too high and the living wages are given too easily.  They have to work to get money, so why don't the refugees?
Here is a third example of how you can detect disinformation in such articles: refugees and immigrants often live in difficult circumstances.
They have difficulty understanding the language and culture of the country they are staying in.  
Frustrations can in some cases lead to violence.
People generalize this violence, exhibited by a few, to the entire group of refugees.  They perceive all refugees as violent and dangerous, and therefore as not wanted in their country.
Answer with a simple yes or no, yes if you think the article contains disinformation, no if you think the article does not contain disinformation.  
Do not give any further explanation or justification.""",
       }
    ),
]


class RowResult(BaseModel):
    url: str
    invalid: bool
    y: int
    y_hat: int

class ModelResult(BaseModel):
    model_input: ModelInput
    row_results: Dict[PromptType, List[RowResult]]

def process_model(model_input: ModelInput, database: str):

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using device {device}", flush=True)

    llm_model = pipeline("text-generation", model=model_input.model_name, device=device)

    row_results = {}
    for prompt_type, prompt in model_input.prompts.items():
        print(f"processing prompt type {prompt_type} for model {model_input.model_name}", flush=True)
        row_results_for_prompt = []
        with sqlite3.connect(database) as conn:
            for row in conn.execute(f"select translated_text, disinformation, url from articles limit 5"):
                text = row[0]
                text = re.sub(r'\s+', ' ', text)
                ground_truth = 1 if row[1] == 'y' else 0
                url = row[2]

                print(f"processing text {text[:100]}", flush=True)
                result = llm_model(generate_messages(prompt, text))

                print(f"result of LLM is {result}, ground truth is {row[1]}", flush=True)
                if not check_result_validity(result):
                    row_results_for_prompt.append(RowResult(url=url, invalid=True, y=ground_truth, y_hat=False))
                else:
                    row_results_for_prompt.append(RowResult(url=url, invalid=False, y=ground_truth, y_hat=get_y_hat(result)))
        row_results[prompt_type] = row_results_for_prompt

    return ModelResult(model_input=model_input, row_results=row_results)

def check_result_validity(result):
    try:
        return result[0]['generated_text'][2]['content'] is not None
    except (IndexError, KeyError, TypeError):
        return False

def get_y_hat(result) -> int:
    try:
        return 1 if result[0]['generated_text'][2]['content'].strip().lower() == 'yes' else 0
    except (IndexError, KeyError, TypeError):
        return False

def generate_messages(prompt: str, text:str):
    return [
            {"role": "system", "content": prompt},
        # we beperken even tot de eerste 2000 woorden
            {"role": "user", "content": " ".join(text.split(" ")[:2000])},
        ]

def sanitize_filename(filename: str) -> str:
    return re.sub(r'[\/:*?"<>|]', '_', filename)

def rq1(database: str):
    for model in model_inputs:
        print(f"processing model {model.model_name}", flush=True)
        model_result = process_model(model, database)
        sanitized_model = sanitize_filename(model.model_name)
        with open(f"rq1_{sanitized_model}.json", "w") as f:
            f.write(model_result.model_dump_json(indent=2))

        for prompt_type, results in model_result.row_results.items():
            print(f"{prompt_type}: Number of invalid results {len([i for i in results if i.invalid])}", flush=True)
            ys = [i.y for i in results]
            y_hats = [i.y_hat for i in results]

            tn, fp, fn, tp = confusion_matrix(ys, y_hats, labels=[0, 1]).ravel()

            print(f"{prompt_type}: True Negatives: {tn}", flush=True)
            print(f"{prompt_type}: False Positives: {fp}", flush=True)
            print(f"{prompt_type}: False Negatives: {fn}", flush=True)
            print(f"{prompt_type}: True Positives: {tp}", flush=True)

            # Calculate metrics
            accuracy = accuracy_score(ys, y_hats)
            precision = precision_score(ys, y_hats)  # Default: binary classification
            recall = recall_score(ys, y_hats)
            f1 = f1_score(ys, y_hats)

            # Print the results
            print(f"{prompt_type}: Accuracy: {accuracy:.2f}", flush=True)
            print(f"{prompt_type}: Precision: {precision:.2f}", flush=True)
            print(f"{prompt_type}: Recall: {recall:.2f}", flush=True)
            print(f"{prompt_type}: F1-score: {f1:.2f}", flush=True)

            # More detailed report
            print("{prompt_type}: Classification Report:\n", classification_report(ys, y_hats, labels=[0, 1], zero_division="warn"), flush=True)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : rq1.py <combined-db.sqlite>")
    rq1(sys.argv[1])