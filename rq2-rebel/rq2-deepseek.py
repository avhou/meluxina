import json
import os
import sqlite3
from datetime import datetime
import argparse

import torch
from transformers import pipeline
from typing import List

from models import Groupings, PromptTemplates, PromptTemplate, ClassificationOutput, ModelResults, ModelResult

print(f"found HUGGINGFACE_HUB_CACHE : {os.environ.get('HUGGINGFACE_HUB_CACHE')}", flush=True)
print(f"found HF_HOME : {os.environ.get('HF_HOME')}", flush=True)
print(f"found HUGGINGFACEHUB_API_TOKEN : {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}", flush=True)

# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
MODEL_NAME = "microsoft/phi-4"
OUTPUT_FILE = "rq2-results-deepseek.json"
PROGRESS_FILE = "rq2-results-progress-deepseek.json"
MODEL_PARAMS = {}


def create_model(model_name: str, model_params: dict):
    print(f"""Starting model load at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

    model = pipeline(
        "text-generation",
        model=model_name,
        # model_kwargs={"torch_dtype": torch.bfloat16, "trust_remote_code": True},
        model_kwargs={"torch_dtype": torch.bfloat16},
        token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        device_map="auto",
    )

    print(f"""Done loading model at {datetime.now().strftime("%H:%M:%S")}""", flush=True)
    return model


def generate_messages(prompt: str, text: str):
    # default pipeline kan hier mee om, non default pipeline echter niet
    data = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    return data


def generate_prompt(prompt_template: PromptTemplate, group_by: Groupings) -> str:
    return f"""
[Context]
You are provided with the following context to assist you in your task:
--- 
{prompt_template.get_context(group_by)}
---

[Instructions]
Follow these instructions carefully.
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
Use the provided context to guide your classification.
Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
Do not give any further explanation or justification. 
Keep your reasoning output to a minimum.  
Be as succinct as possible.  
The only relevant output is your final classification.  
Generate your output in JSON format.  
The output should conform to this JSON schema : {ClassificationOutput.model_json_schema()}.
    """


def get_prompt_template(prompts: str, group_by: Groupings) -> List[PromptTemplate]:
    with open(prompts, "r") as f:
        return PromptTemplates.model_validate_json(f.read()).get_templates(group_by)


def read_results(file: str) -> ModelResults:
    with open(file, "r") as f:
        return ModelResults.model_validate_json(f.read())


def write_results(file: str, results: ModelResults):
    with open(file, "w") as f:
        return f.write(results.model_dump_json(indent=2))


def process_prompts(prompts: str, group_by: Groupings):
    print(f"processing prompts file {prompts}, group_by {group_by}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using device {device}", flush=True)

    prompt_templates = get_prompt_template(prompts, group_by)
    model = create_model(MODEL_NAME, MODEL_PARAMS)

    row_results: List[ModelResult] = []
    existing_row_results: List[ModelResult] = []
    if os.path.exists(PROGRESS_FILE):
        try:
            existing_row_results = read_results(PROGRESS_FILE).results
        except Exception as e:
            print(f"could not parse existing file {PROGRESS_FILE}, error {e}", flush=True)
            existing_row_results = []

    for i, prompt_template in enumerate(prompt_templates):
        print(f"({i + 1}/{len(prompt_templates)}): processing url {prompt_template.url}", flush=True)

        y: int = 1 if prompt_template.ground_truth_disinformation == "y" else 0

        existing_row_for_url = next((x for x in existing_row_results if x.url == prompt_template.url), None)
        if existing_row_for_url is not None:
            print(f"skipping url {prompt_template.url} as it is already processed", flush=True)
            row_results.append(existing_row_for_url)
            write_results(PROGRESS_FILE, ModelResults(results=row_results))
            continue

        messages = generate_messages(
            generate_prompt(prompt_template, group_by),
            prompt_template.article_text,
        )
        try:
            outputs = model(messages, max_new_tokens=2500)
            result = outputs[0]["generated_text"][-1]

            print(f"result of LLM is {result}, ground truth is {prompt_template.ground_truth_disinformation}", flush=True)
            try:
                string_representation = json.dumps(result)
            except:
                string_representation = str(result)
            row_results.append(ModelResult(url=prompt_template.url, result=string_representation, y=y))
        except Exception as e:
            row_results.append(ModelResult(url=prompt_template.url, result=f"an error occurrred {e}", y=y))

        write_results(PROGRESS_FILE, ModelResults(results=row_results))
    write_results(OUTPUT_FILE, ModelResults(results=row_results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2")

    parser.add_argument("--prompts", type=str, required=True, help="Path to the prompts file")
    parser.add_argument(
        "--group-by",
        type=str,
        required=True,
        choices=["article", "chunk", "triple"],
        help="The grouping to use to create additional context",
    )
    args = parser.parse_args()

    process_prompts(args.prompts, args.group_by)

    print("done")
