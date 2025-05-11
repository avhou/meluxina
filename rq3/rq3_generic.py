import json
import os
import argparse

import torch
from typing import List, Callable

from models import PromptTemplates, PromptTemplate, ModelResult, ModelResults
from transformers import pipeline
from datetime import datetime


def create_model(model_name: str, model_params: dict):
    print(f"""Starting model load at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

    model = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        device_map="auto",
    )

    print(f"""Done loading model at {datetime.now().strftime("%H:%M:%S")}""", flush=True)
    return model


def output_file(model: str, threaded: bool) -> str:
    return f"rq3-results-group-by-{model}.json" if not threaded else f"rq3-results-group-by-{model}-threaded.json"


def progress_file(model: str, threaded: bool) -> str:
    return f"rq3-progress-group-by-{model}.json" if not threaded else f"rq3-progress-group-by-{model}-threaded.json"


def generate_messages(prompt: str, text: str):
    # default pipeline kan hier mee om, non default pipeline echter niet
    data = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    return data


def generate_prompt(prompt_template: PromptTemplate, instructions: str) -> str:
    return f"""
[Context]
You are provided with the following context to assist you in your task:
--- 
{prompt_template.get_context()}
---

[Instructions]
{instructions}
"""


def get_prompt_template(prompts: str) -> List[PromptTemplate]:
    with open(prompts, "r") as f:
        return PromptTemplates.model_validate_json(f.read()).templates


def read_results(file: str) -> ModelResults:
    with open(file, "r") as f:
        return ModelResults.model_validate_json(f.read())


def write_results(file: str, results: ModelResults):
    with open(file, "w") as f:
        return f.write(results.model_dump_json(indent=2))


def process_prompts(prompts: str, model_generator: Callable[[], any], short_model_name: str, instructions: str, max_words: int = 5000):
    is_threaded = "threaded" in prompts
    print(f"processing prompts file {prompts}, is_threaded {is_threaded}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using device {device}", flush=True)

    prompt_templates = get_prompt_template(prompts)
    model = model_generator()

    row_results: List[ModelResult] = []
    existing_row_results: List[ModelResult] = []
    if os.path.exists(progress_file(short_model_name, is_threaded)):
        try:
            existing_row_results = read_results(progress_file(short_model_name, is_threaded)).results
        except Exception as e:
            print(f"could not parse existing file {progress_file(short_model_name, is_threaded)}, error {e}", flush=True)
            existing_row_results = []

    for i, prompt_template in enumerate(prompt_templates):
        print(f"({i + 1}/{len(prompt_templates)}): processing url {prompt_template.url}", flush=True)

        y: int = 1 if prompt_template.ground_truth_disinformation == "y" else 0

        existing_row_for_url = next((x for x in existing_row_results if x.url == prompt_template.url), None)
        if existing_row_for_url is not None:
            print(f"skipping url {prompt_template.url} as it is already processed", flush=True)
            row_results.append(existing_row_for_url)
            write_results(progress_file(short_model_name, is_threaded), ModelResults(results=row_results))
            continue

        # give context and article equal number of words for now
        messages = generate_messages(
            generate_prompt(prompt_template, instructions),
            prompt_template.get_article_text(int(max_words / 2)),
        )
        print(messages, flush=True)
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

        write_results(progress_file(short_model_name, is_threaded), ModelResults(results=row_results))
    write_results(output_file(short_model_name, is_threaded), ModelResults(results=row_results))


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ3")

    parser.add_argument("--prompts", type=str, required=True, help="Path to the prompts file")
    parser.add_argument("--max-words", type=int, required=False, help="Max number of words to use", default=10_000)
    return parser
