import transformers
from models import *

from datetime import datetime
import sqlite3
import sys
import torch
import os
import re


print(f"found HUGGINGFACE_HUB_CACHE : {os.environ.get('HUGGINGFACE_HUB_CACHE')}", flush=True)
print(f"found HF_HOME : {os.environ.get('HF_HOME')}", flush=True)
print(f"found HUGGINGFACEHUB_API_TOKEN : {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}", flush=True)

# deepseek is hier bijzonder slecht in.  blijft hangen op eerste ttl vraag
# daarom uit de lijst gehaald

# ModelInput(
#     model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
#     model_params={},
# ),

model_inputs = [
    # ModelInput(
    #     model_name="meta-llama/Llama-3.3-70B-Instruct",
    #     model_params={},
    # ),
    ModelInput(
        model_name="microsoft/phi-4",
        model_params={},
    ),
]

def process_model(model_input: ModelInput, database: str, condensed_ontology: str):

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using device {device}", flush=True)

    print(f"""Starting model load at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_input.model_name,
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    print(f"""Done loading model at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

    row_results = []
    print(f"processing model {model_input.model_name}", flush=True)
    with sqlite3.connect(database) as conn:
        for row in conn.execute(f"select translated_text, disinformation, url from articles limit 1"):
            text = row[0]
            text = re.sub(r'\s+', ' ', text)
            ground_truth = 1 if row[1] == 'y' else 0
            url = row[2]


            try:
                # print(f"processing text to TTL for {text[:100]}", flush=True)
                # messages = [
                #     {"role": "system", "content": """You are an expert AI system that specializes in named entity recognition and knowledge graph extraction.
                #     You will receive two inputs: an article text and an ontology describing the most important concepts and relations of the domain we study.
                #     For ease, the ontology will be summarized in subject ~ predicate ~ object triples.
                #     Each triple is separated by a newline character, and subject, predicate and object are separated by a tilde character.
                #     Your task is to analyse the article text, identify and extract the relevant entities and their attributes and relationships from the article text, relate them as much as possible to the ontology,
                #     and output a knowledge graph in Turtle (TTL) format composed by RDF triples (subject, predicate, object).
                #     Only output the TTL-formatted knowledge graph and do not include any explanations.
                #     Do not output the condensed triple format, but the full Turtle format.
                #     Be as succinct as possible and only include the most relevant information.
                #     Try to minimize the number of triples in the output, while keeping the most relevant information.
                #     Make sure to list just one concept per subject, predicate or object.  If a subject, predicate or object contains multiple concepts, split them into separate triples.
                #     """},
                #     {"role": "user", "content": f"This is the ontology:\n{condensed_ontology}\nThis is the article text: {text}"},
                # ]
                #
                # outputs = pipeline(messages, max_new_tokens=4096)
                # ttl = outputs[0]["generated_text"][-1]["content"]
                ttl = ''

                print(f"processing text to JSON for {text[:100]}", flush=True)
                messages = [
                    {"role": "system", "content": f"""You are an expert AI system that specializes in named entity recognition and knowledge graph extraction. 
                    You will receive two inputs: an article text and an ontology describing the most important concepts and relations of the domain we study.
                    The ontology will be summarized in subject ~ predicate ~ object triples.  
                    Each triple is separated by a newline character, and subject, predicate and object are separated by a tilde character.
                    Your task is to analyse the article text, identify and extract the relevant entities and their attributes and relationships from the article text, and output a knowledge graph.
                    You may refer to the ontology and relate the entities you extract from the article to that ontology, if possible.  
                    However, the main goal is to extract relevant info from the article text, and the ontology is only there to help you.
                    You will output triples (subject, predicate, object) in JSON format.   
                    The output should conform to this JSON schema : {Output.model_json_schema()}.  
                    Only output the JSON-formatted knowledge graph and do not include any explanations.  
                    Be as succinct as possible and only include the most relevant information.
                    Try to minimize the number of triples in the output, while keeping the most relevant information.
                    Make sure to list just one concept per subject, predicate or object.  If a subject, predicate or object contains multiple concepts, split them into separate triples.
                    """},
                    {"role": "user", "content": f"This is the ontology:\n{condensed_ontology}\nThis is the article text: {text}"},
                ]

                outputs = pipeline(messages, max_new_tokens=4096)
                json = outputs[0]["generated_text"][-1]["content"]

                row_results.append(RowResult(url=url, valid=True, result_ttl=ttl, result_json=json, y=ground_truth))

            except Exception as e:
                row_results.append(RowResult(url=url, valid=False, result_ttl=f"got exception {e}", result_json=f"got exception {e}", y=ground_truth))

            sanitized_model = sanitize_filename(model_input.model_name)
            with open(f"rq2_refined_{sanitized_model}.json", "w") as f:
                f.write(ModelResult(model_input=model_input, row_results=row_results).model_dump_json(indent=2))

    return ModelResult(model_input=model_input, row_results=row_results)



def rq2_refine(database: str, condensed_ontology: str):
    with open(condensed_ontology, "r") as f:
        condensed_ontology = f.read()
        for model in model_inputs:
            print(f"processing model {model.model_name}", flush=True)
            model_result = process_model(model, database, condensed_ontology)
            sanitized_model = sanitize_filename(model.model_name)
            with open(f"rq2_refined_{sanitized_model}.json", "w") as f:
                f.write(model_result.model_dump_json(indent=2))
    print(f"Done.")

if __name__ == "__main__":
    if len(sys.argv) <= 2:
        raise RuntimeError("usage : rq2_refine_adhv_ontology.py <combined-db.sqlite> <condensed_ontology.txt>")
    rq2_refine(sys.argv[1], sys.argv[2])
