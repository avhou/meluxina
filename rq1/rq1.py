import sqlite3
import sys
from transformers import pipeline
import torch
from typing import List, Dict, Literal
from pydantic import BaseModel
import re
from sklearn.metrics import confusion_matrix

PromptType = Literal["zero-shot", "one-shot", "few-shot"]

models = ["meta-llama/Llama-3.2-3B-Instruct"]
prompts = ["""You are a research assistant that tries to detect disinformation in articles.  
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation 
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).   
Answer with a simple yes or no, yes if you think the article contains disinformation, no if you think the article does not contain disinformation.  
Do not give any further explanation or justification."""
           ]
prompt_types: List[PromptType] = ["zero-shot"]


class RowResult(BaseModel):
    url: str
    invalid: bool
    y: int
    y_hat: int

class ModelResult(BaseModel):
    model: str
    row_results: Dict[PromptType, List[RowResult]]

def process_model(model: str, database: str):

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using device {device}")

    row_results = {}
    for prompt, prompt_type in zip(prompts, prompt_types):
        chatbot = pipeline("text-generation", model=model, device=device)
        row_results_for_prompt = []
        with sqlite3.connect(database) as conn:
            for row in conn.execute(f"select translated_text, disinformation, url from articles"):
                text = row[0]
                ground_truth = 1 if row[1] == 'y' else 0
                url = row[2]

                print(f"processing text {row[0][:100]}")
                result = chatbot(generate_messages(prompt, row[0]))

                print(f"result of LLM is {result[0]['generated_text'][2]['content']}, ground truth is {row[1]}")
                if not check_result_validity(result):
                    row_results_for_prompt.append(RowResult(url=url, invalid=True, y=ground_truth, y_hat=False))
                else:
                    row_results_for_prompt.append(RowResult(url=url, invalid=False, y=ground_truth, y_hat=get_y_hat(result)))
        row_results[prompt_type] = row_results_for_prompt

    return ModelResult(model=model, row_results=row_results)

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
            {"role": "user", "content": text},
        ]

def sanitize_filename(filename: str) -> str:
    return re.sub(r'[\/:*?"<>|]', '_', filename)

def rq1(database: str):
    for model in models:
        print(f"processing model {model}")
        model_result = process_model(model, database)
        sanitized_model = sanitize_filename(model)
        with open(f"rq1_{sanitized_model}.json", "w") as f:
            f.write(model_result.model_dump_json(indent=2))

        tn, fp, fn, tp = confusion_matrix([i.y for i in model_result.row_results["zero-shot"]], [i.y_hat for i in model_result.row_results["zero-shot"]]).ravel()

        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : rq1.py <combined-db.sqlite>")
    rq1(sys.argv[1])