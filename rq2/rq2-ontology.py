from pathlib import Path

import transformers
from models import *
from itertools import islice

from datetime import datetime
import sys
import torch
import os


print(f"found HUGGINGFACE_HUB_CACHE : {os.environ.get('HUGGINGFACE_HUB_CACHE')}", flush=True)
print(f"found HF_HOME : {os.environ.get('HF_HOME')}", flush=True)
print(f"found HUGGINGFACEHUB_API_TOKEN : {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}", flush=True)


def create_pipeline():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using device {device}", flush=True)

    print(f"""Starting model load at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/phi-4",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    print(f"""Done loading model at {datetime.now().strftime("%H:%M:%S")}""", flush=True)
    return pipeline

def generate_json_ontology(text: str, pipeline) -> str:
    try:
        print(f"processing ontology (JSON)", flush=True)

        messages = [
            {"role": "system", "content": f"""You are an expert AI system that specializes in named entity recognition and knowledge graph extraction. 
            You will be given a knowledge graph in JSON format composed by RDF triples (subject, predicate, object).
            The input will conform to this JSON schema : {Output.model_json_schema()}.
            This knowledge graph was created by merging the JSON-formatted knowledge graphs of multiple input texts.
            Therefore, it may contain duplicate entitities under a different name or URI, but with the same meaning.
            It is your task to analyse the knowledge graph, merge similar or identical entities, simplify the knowledge graph as much as possible, renaming very similar concepts to keep only one singe concept,  and output an ontology as a new knowledge graph in JSON format.
            The input triples are sorted by subject first, predicate second and object third, so it will be easy to spot identical subjects (and predicates).
            Try to minimize the number of triples in the output ontology, while keeping the most relevant information.
            The output should conform to this JSON schema : {Output.model_json_schema()}.  
            Only output the JSON-formatted ontology and do not include any explanations.
            Be as succinct as possible and only include the most relevant information.
            """},
            {"role": "user", "content": text},
        ]

        outputs = pipeline(messages, max_new_tokens=4096)
        return outputs[0]["generated_text"][-1]["content"]

    except Exception as e:
        return f"exception: {e}"


def generate_ttl_ontology(text: str, pipeline) -> str:
    try:
        print(f"processing ontology (TTL)", flush=True)

        messages = [
            {"role": "system", "content": f"""You are an expert AI system that specializes in named entity recognition and knowledge graph extraction. 
                You will be given a knowledge graph in Turtle (TTL) format composed by RDF triples (subject, predicate, object).
                This knowledge graph was created by concatenating the TTL-formatted knowledge graphs of multiple input texts.
                Therefore, it may contain duplicate entitities under a different name or URI, but with the same meaning.
                It is your task to analyse the knowledge graph, merge similar or identical entities, simplify the knowledge graph as much as possible, and output an ontology as a new knowledge graph in Turtle (TTL) format.
                You are allowed to remove triples, subject, objects or predicates if you can replace them with fewer triples, for example triples that capture a more general concept.
                Only output the TTL-formatted ontology and do not include any explanations.
                Be as succinct as possible and only include the most relevant information.
                However, make sure to list just one concept per subject, predicate or object.  If a subject, predicate or object contains multiple concepts, split them into separate triples.
            """},
            {"role": "user", "content": text},
        ]

        outputs = pipeline(messages, max_new_tokens=2048)
        return outputs[0]["generated_text"][-1]["content"]

    except Exception as e:
        return f"exception: {e}"


def grouper_with_index(iterable, n):
    it = iter(iterable)
    for index, chunk in enumerate(iter(lambda: list(islice(it, n)), [])):
        yield index, chunk

def rq2_ontology(input_folder: str):
    print(f"processing input_folder {input_folder}")
    pipeline = create_pipeline()
    for input_file in Path(input_folder).rglob("*_combined.json"):
        print(f"processing input_file {input_file}")
        with open(input_file, "r") as f:
            content = f.read()
            model_result = Output.model_validate_json(content)

            if model_result is None:
                print(f"Could not parse input file {input_file}")
                continue

            for group_index, group in grouper_with_index(model_result.triples, 250):
                print(f"processing group {group_index + 1} of {input_file}, contains {len(group)} triples, total nr of triples is {len(model_result.triples)}", flush=True)
                output = generate_json_ontology(Output(triples=group).model_dump_json(indent=0), pipeline)
                with open(f"{input_file.stem}_ontology_{group_index + 1}.json", "w") as f:
                    f.write(remove_markdown(output))


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : rq2.py <input-folder>")
    rq2_ontology(sys.argv[1])
