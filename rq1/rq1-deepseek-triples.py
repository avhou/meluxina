import sqlite3
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from datetime import datetime
from models import *
import json

print(
    f"found HUGGINGFACE_HUB_CACHE : {os.environ.get('HUGGINGFACE_HUB_CACHE')}",
    flush=True,
)
print(f"found HF_HOME : {os.environ.get('HF_HOME')}", flush=True)
print(
    f"found HUGGINGFACEHUB_API_TOKEN : {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}",
    flush=True,
)


model_inputs = [
    ModelInput(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        model_params={"trust_remote_code": True},
        prompt_generation=lambda prompt, text: generate_messages(prompt, text),
        model_creation=lambda input: create_model(input),
        prompts={
            "zero-shot": f"""You are a research assistant that tries to detect disinformation in articles.
    A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
    (that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
    The article is summarized in subject ~ predicate ~ object triples.  You will not have access to the original article, only to the triples.
    Each triple is separated by a newline character, and subject, predicate and object are separated by a tilde character.
    Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
    Do not give any further explanation or justification. Keep your reasoning output to a minimum.  There is no need to output your reasoning for each triple.  Be as succinct as possible.  The only relevant output is your final classification.  Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
            "one-shot": f"""You are a research assistant that tries to detect disinformation in articles.
    A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
    (that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
    The article is summarized in subject ~ predicate ~ object triples.  You will not have access to the original article, only to the triples.
    Each triple is separated by a newline character, and subject, predicate and object are separated by a tilde character.
    Here is one example of how you can detect disinformation in such articles: 
    war refugees ~ flee to ~ the Netherlands
    war refugees ~ get immediate access to ~ social housing infrastructure
    Dutch citizens ~ need access to ~ social housing infrastructure
    Dutch citizens ~ must wait ~ many years
    war refugees ~ get ~ preferential treatment
    Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
    Do not give any further explanation or justification. Keep your reasoning output to a minimum.  There is no need to output your reasoning for each triple.  Be as succinct as possible.  The only relevant output is your final classification.  Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
            "few-shot": f"""You are a research assistant that tries to detect disinformation in articles.
    A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
    (that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
    The article is summarized in subject ~ predicate ~ object triples.  You will not have access to the original article, only to the triples.
    Each triple is separated by a newline character, and subject, predicate and object are separated by a tilde character.
    Here is a first example of how you can detect disinformation in such articles:
    war refugees ~ flee to ~ the Netherlands
    war refugees ~ get immediate access to ~ social housing infrastructure
    Dutch citizens ~ need access to ~ social housing infrastructure
    Dutch citizens ~ must wait ~ many years
    war refugees ~ get ~ preferential treatment
    Here is a second example of how you can detect disinformation in such articles: 
    war refugees ~ receive ~ living wage
    more war refugees ~ increase ~ costs of living wage
    cost of living wage ~ is ~ too high 
    living wage ~ given ~ too easily
    dutch citizens ~ must work ~ for money
    war refugees ~ do not have to work ~ for money
    Here is a third example of how you can detect disinformation in such articles: 
    war refugees ~ live in ~ difficult circumstances
    war refugees ~ don't understand ~ language of host country
    war refugees ~ don't understand ~ culture of host country
    some war refugees ~ experience ~ frustration
    frustration ~ can lead to ~ violence
    all war refugees ~ lead to ~ violence
    Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
    Do not give any further explanation or justification. Keep your reasoning output to a minimum.  There is no need to output your reasoning for each triple.  Be as succinct as possible.  The only relevant output is your final classification.  Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
        },
    ),
]


def create_model(model_input: ModelInput):
    print(
        f"""Starting model load at {datetime.now().strftime("%H:%M:%S")}""", flush=True
    )

    model = pipeline(
        "text-generation",
        model=model_input.model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        device_map="auto",
    )

    print(
        f"""Done loading model at {datetime.now().strftime("%H:%M:%S")}""", flush=True
    )
    return model


def process_model(
    model_input: ModelInput,
    database: str,
    triple_map: Dict[str, List[Triple]],
    fallback_triple_map: Dict[str, List[Triple]],
    triple_generation_model: str,
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"using device {device}", flush=True)

    llm_model = model_input.model_creation(model_input)

    sanitized_model = sanitize_filename(model_input.model_name)
    row_results = {}
    for prompt_type, prompt in model_input.prompts.items():
        progress_file = f"rq1_triples_{sanitized_model}_{prompt_type}_generated_by_{triple_generation_model}.json"

        print(
            f"processing prompt type {prompt_type} for model {model_input.model_name}",
            flush=True,
        )
        row_results_for_prompt = []
        existing_row_results_for_prompt = []
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                try:
                    content = f.read()
                    model_result = ModelResult.model_validate_json(content)
                    existing_row_results_for_prompt = model_result.row_results[
                        prompt_type
                    ]
                except Exception as e:
                    print(
                        f"could not parse existing file {progress_file}, error {e}",
                        flush=True,
                    )
                    existing_row_results_for_prompt = []

        with sqlite3.connect(database) as conn:
            for row in conn.execute(
                f"select translated_text, disinformation, url from articles limit 1"
            ):
                text = row[0]
                text = re.sub(r"\s+", " ", text)
                ground_truth = 1 if row[1] == "y" else 0
                url = row[2]
                triples_for_url = triple_map.get(url, fallback_triple_map.get(url, []))

                existing_row_for_url = next(
                    (x for x in existing_row_results_for_prompt if x.url == url), None
                )
                if existing_row_for_url is not None:
                    print(f"skipping url {url} as it is already processed", flush=True)
                    row_results_for_prompt.append(existing_row_for_url)
                    with open(progress_file, "w") as f:
                        f.write(
                            ModelResult(
                                model_input=model_input,
                                row_results={prompt_type: row_results_for_prompt},
                            ).model_dump_json(indent=2)
                        )
                    continue

                print(f"processing url {url}", flush=True)

                input_prompt = model_input.prompt_generation(
                    prompt, "\n".join([f"{t}" for t in triples_for_url])
                )
                try:
                    outputs = llm_model(input_prompt, max_new_tokens=1000)
                    result = outputs[0]["generated_text"][-1]

                    print(
                        f"result of LLM is {result}, ground truth is {ground_truth}, raw ground truth is {row[1]}",
                        flush=True,
                    )
                    try:
                        string_representation = json.dumps(result)
                    except:
                        string_representation = str(result)
                    row_results_for_prompt.append(
                        RowResult(
                            url=url,
                            invalid=True,
                            y=ground_truth,
                            y_hat=False,
                            result=string_representation,
                        )
                    )
                except Exception as e:
                    row_results_for_prompt.append(
                        RowResult(
                            url=url,
                            invalid=True,
                            y=ground_truth,
                            y_hat=False,
                            result=f"an error occurred : {e}",
                        )
                    )

                with open(progress_file, "w") as f:
                    f.write(
                        ModelResult(
                            model_input=model_input,
                            row_results={prompt_type: row_results_for_prompt},
                        ).model_dump_json(indent=2)
                    )
        row_results[prompt_type] = row_results_for_prompt

    return ModelResult(model_input=model_input, row_results=row_results)


def check_result_validity(result):
    return False


def get_y_hat(result) -> int:
    return -1


def generate_messages(prompt: str, text: str):
    # default pipeline kan hier mee om, non default pipeline echter niet
    data = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    return data


def rq1(database: str, triple_file: str, fallback_triple_file: Optional[str] = None):
    (triple_generation_model, triple_map) = read_triple_map(triple_file)
    print(f"triple generation model is {triple_generation_model}")
    if fallback_triple_file is not None:
        print(f"using fallback triple file {fallback_triple_file}")
        (_, fallback_triple_map) = read_triple_map(fallback_triple_file)
    else:
        fallback_triple_map = triple_map

    for model in model_inputs:
        print(
            f"processing model {model.model_name} and triple file {triple_file}",
            flush=True,
        )
        model_result = process_model(
            model, database, triple_map, fallback_triple_map, triple_generation_model
        )
        sanitized_model = sanitize_filename(model.model_name)
        with open(
            f"rq1_triples_{sanitized_model}_generated_by_{triple_generation_model}.json",
            "w",
        ) as f:
            f.write(model_result.model_dump_json(indent=2))


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        raise RuntimeError("usage : rq1.py <combined-db.sqlite> <triple-file.json")
    rq1(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
