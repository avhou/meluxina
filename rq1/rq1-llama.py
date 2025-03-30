import sqlite3
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from datetime import datetime
from models import *

print(f"found HUGGINGFACE_HUB_CACHE : {os.environ.get('HUGGINGFACE_HUB_CACHE')}", flush=True)
print(f"found HF_HOME : {os.environ.get('HF_HOME')}", flush=True)
print(f"found HUGGINGFACEHUB_API_TOKEN : {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}", flush=True)


model_inputs = [
        ModelInput(
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            model_params={"trust_remote_code": True},
            prompt_generation=lambda prompt, text: generate_messages(prompt, text),
            model_creation=lambda input: create_model_llama(input),
            prompts={
                "zero-shot": f"""You are a research assistant that tries to detect disinformation in articles.
    A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
    (that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
    Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
    Do not give any further explanation or justification. Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
                "one-shot": f"""You are a research assistant that tries to detect disinformation in articles.
    A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
    (that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
    Here is one example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.
    They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.
    Dutch citizens feel the refugees receive preferential treatment.
    Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
    Do not give any further explanation or justification. Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
                "few-shot": f"""You are a research assistant that tries to detect disinformation in articles.
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
    Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
    Do not give any further explanation or justification. Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
           }
        ),
]


def create_model_llama(model_input: ModelInput):
    print(f"""Starting model load at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

    model = pipeline(
        "text-generation",
        model=model_input.model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    print(f"""Done loading model at {datetime.now().strftime("%H:%M:%S")}""", flush=True)
    return model


def process_model(model_input: ModelInput, database: str):

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using device {device}", flush=True)

    llm_model = model_input.model_creation(model_input)

    sanitized_model = sanitize_filename(model_input.model_name)
    row_results = {}
    for prompt_type, prompt in model_input.prompts.items():
        progress_file = f"rq1_{sanitized_model}_{prompt_type}.json"

        print(f"processing prompt type {prompt_type} for model {model_input.model_name}", flush=True)
        row_results_for_prompt = []
        existing_row_results_for_prompt = []
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                try:
                    content = f.read()
                    model_result = ModelResult.model_validate_json(content)
                    existing_row_results_for_prompt = model_result.row_results[prompt_type]
                except Exception as e:
                    print(f"could not parse existing file {progress_file}, error {e}", flush=True)
                    existing_row_results_for_prompt = []

        with sqlite3.connect(database) as conn:
            for row in conn.execute(f"select translated_text, disinformation, url from articles"):
                text = row[0]
                text = re.sub(r'\s+', ' ', text)
                ground_truth = 1 if row[1] == 'y' else 0
                url = row[2]

                existing_row_for_url = next((x for x in existing_row_results_for_prompt if x.url == url), None)
                if existing_row_for_url is not None:
                    print(f"skipping url {url} as it is already processed", flush=True)
                    row_results_for_prompt.append(existing_row_for_url)
                    with open(progress_file, "w") as f:
                        f.write(ModelResult(model_input=model_input, row_results={prompt_type: row_results_for_prompt}).model_dump_json(indent=2))
                    continue

                print(f"processing text {text[:100]}", flush=True)

                input_prompt = model_input.prompt_generation(prompt, text)
                try:
                    outputs = llm_model(input_prompt, max_new_tokens=1000)
                    result = outputs[0]["generated_text"][-1]

                    print(f"result of LLM is {result}, ground truth is {row[1]}", flush=True)
                    if not check_result_validity(result):
                        row_results_for_prompt.append(RowResult(url=url, invalid=True, y=ground_truth, y_hat=False, result=result))
                    else:
                        row_results_for_prompt.append(RowResult(url=url, invalid=False, y=ground_truth, y_hat=get_y_hat(result), result=result))
                except Exception as e:
                    row_results_for_prompt.append(
                        RowResult(url=url, invalid=True, y=ground_truth, y_hat=False, result=f"an error occurred : {e}"))

                with open(progress_file, "w") as f:
                    f.write(ModelResult(model_input=model_input, row_results={prompt_type: row_results_for_prompt}).model_dump_json(indent=2))
        row_results[prompt_type] = row_results_for_prompt

    return ModelResult(model_input=model_input, row_results=row_results)


def check_result_validity(result):
    return False

def get_y_hat(result) -> int:
    return -1

def generate_messages(prompt: str, text:str):
    # default pipeline kan hier mee om, non default pipeline echter niet
    data = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    return data

def rq1(database: str):
    for model in model_inputs:
        print(f"processing model {model.model_name}", flush=True)
        model_result = process_model(model, database)
        sanitized_model = sanitize_filename(model.model_name)
        with open(f"rq1_{sanitized_model}.json", "w") as f:
            f.write(model_result.model_dump_json(indent=2))

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : rq1.py <combined-db.sqlite>")
    rq1(sys.argv[1])
