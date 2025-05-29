import transformers
from models import *

from datetime import datetime
import sqlite3
import sys
import torch
import os
import re


print(
    f"found HUGGINGFACE_HUB_CACHE : {os.environ.get('HUGGINGFACE_HUB_CACHE')}",
    flush=True,
)
print(f"found HF_HOME : {os.environ.get('HF_HOME')}", flush=True)
print(
    f"found HUGGINGFACEHUB_API_TOKEN : {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}",
    flush=True,
)

# deepseek is hier bijzonder slecht in.  blijft hangen op eerste ttl vraag
# daarom uit de lijst gehaald

# ModelInput(
#     model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
#     model_params={},
# ),

model_inputs = [
    ModelInput(
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        model_params={},
    ),
    # ModelInput(
    #     model_name="microsoft/phi-4",
    #     model_params={},
    # ),
]


def process_model(model_input: ModelInput, database: str):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"using device {device}", flush=True)

    print(
        f"""Starting model load at {datetime.now().strftime("%H:%M:%S")}""", flush=True
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_input.model_name,
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    print(
        f"""Done loading model at {datetime.now().strftime("%H:%M:%S")}""", flush=True
    )

    row_results = []
    existing_row_results = []
    sanitized_model = sanitize_filename(model_input.model_name)
    progress_file = f"rq2_threaded_{sanitized_model}.json"

    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            try:
                content = f.read()
                model_result = ModelResult.model_validate_json(content)
                existing_row_results = model_result.row_results
            except Exception as e:
                print(
                    f"could not parse existing file {progress_file}, error {e}",
                    flush=True,
                )
                existing_row_results = []

    print(f"processing model {model_input.model_name}", flush=True)
    with sqlite3.connect(database) as conn:
        total = conn.execute("select count(*) from articles").fetchone()[0]
        r = 0
        for row in conn.execute(
            f"select translated_text, disinformation, url from articles"
        ):
            text = row[0]
            text = re.sub(r"\s+", " ", text)
            ground_truth = 1 if row[1] == "y" else 0
            url = row[2]
            r = r + 1

            existing_row_for_url = next(
                (x for x in existing_row_results if x.url == url), None
            )
            if existing_row_for_url is not None:
                print(f"skipping url {url} as it is already processed", flush=True)
                row_results.append(existing_row_for_url)
                with open(progress_file, "w") as f:
                    f.write(
                        ModelResult(
                            model_input=model_input, row_results=row_results
                        ).model_dump_json(indent=2)
                    )
                continue

            try:
                # print(f"processing text to TTL for {text[:100]}", flush=True)
                # messages = [
                #     {"role": "system", "content": """You are an expert AI system that specializes in named entity recognition and knowledge graph extraction.
                #     Your task is to analyse any provided input text, identify and extract the relevant entities and their attributes and relationships,
                #     and output a knowledge graph in Turtle (TTL) format composed by RDF triples (subject, predicate, object).
                #     Only output the TTL-formatted knowledge graph and do not include any explanations.
                #     Be as succinct as possible and only include the most relevant information.
                #     However, make sure to list just one concept per subject, predicate or object.  If a subject, predicate or object contains multiple concepts, split them into separate triples.
                #     """},
                #     {"role": "user", "content": text},
                # ]
                #
                # outputs = pipeline(messages, max_new_tokens=2048)
                # ttl = outputs[0]["generated_text"][-1]["content"]

                print(
                    f"row {r}/{total}, start at {datetime.now().strftime('%H:%M:%S')}, processing text to JSON for {text[:100]}",
                    flush=True,
                )
                messages = [
                    {
                        "role": "system",
                        "content": f"""You are an expert AI system that specializes in named entity recognition and knowledge graph extraction. 
                    Your task is to analyse any provided input text, identify and extract the relevant entities and their attributes and relationships, 
                    and output a knowledge graph.  
                    You will output triples (subject, predicate, object) in JSON format.   
                    The output should conform to this JSON schema : {Output.model_json_schema()}.  
                    Only output the JSON-formatted knowledge graph and do not include any explanations.  
                    Be as succinct as possible and only include the most relevant information.
                    However, make sure to list just one concept per subject, predicate or object.  If a subject, predicate or object contains multiple concepts, split them into separate triples.
                    """,
                    },
                    {"role": "user", "content": text},
                ]

                outputs = pipeline(messages, max_new_tokens=2048)
                json = outputs[0]["generated_text"][-1]["content"]
                print(
                    f"""Generated triples at {datetime.now().strftime("%H:%M:%S")}""",
                    flush=True,
                )

                row_results.append(
                    RowResult(
                        url=url,
                        valid=True,
                        result_ttl="",
                        result_json=json,
                        y=ground_truth,
                    )
                )

            except Exception as e:
                row_results.append(
                    RowResult(
                        url=url,
                        valid=False,
                        result_ttl=f"got exception {e}",
                        result_json=f"got exception {e}",
                        y=ground_truth,
                    )
                )

            sanitized_model = sanitize_filename(model_input.model_name)
            with open(progress_file, "w") as f:
                f.write(
                    ModelResult(
                        model_input=model_input, row_results=row_results
                    ).model_dump_json(indent=2)
                )

    return ModelResult(model_input=model_input, row_results=row_results)


def rq2(database: str):
    for model in model_inputs:
        print(f"processing model {model.model_name}", flush=True)
        model_result = process_model(model, database)
        sanitized_model = sanitize_filename(model.model_name)
        with open(f"rq2_threaded_{sanitized_model}.json", "w") as f:
            f.write(model_result.model_dump_json(indent=2))
    print(f"Done.")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : rq2.py <combined-db.sqlite>")
    rq2(sys.argv[1])
