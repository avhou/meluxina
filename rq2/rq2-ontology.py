import transformers
from models import *

from datetime import datetime
import sys
import torch
import os


print(f"found HUGGINGFACE_HUB_CACHE : {os.environ.get('HUGGINGFACE_HUB_CACHE')}", flush=True)
print(f"found HF_HOME : {os.environ.get('HF_HOME')}", flush=True)
print(f"found HUGGINGFACEHUB_API_TOKEN : {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}", flush=True)


def generate_ontology(prefix: str, is_json: bool) -> str:

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using device {device}", flush=True)

    print(f"""Starting model load at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

    pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Llama-3.3-70B-Instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    print(f"""Done loading model at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

    try:
        print(f"processing ontology for prefix {prefix}", flush=True)

        if not is_json:
            text = open(f"{prefix}.ttl", "r").read()
            messages = [
                {"role": "system", "content": """You are an expert AI system that specializes in named entity recognition and knowledge graph extraction. 
                You will be given a knowledge graph in Turtle (TTL) format composed by RDF triples (subject, predicate, object).
                This knowledge graph was created by concatenating the TTL-formatted knowledge graphs of multiple input texts.
                Therefore, it may contain duplicate entitities under a different name or URI, but with the same meaning.
                It is your task to analyse the knowledge graph, merge similar or identical entities, simplify the knowledge graph as much as possible, and output an ontology as a new knowledge graph in Turtle (TTL) format.
                Only output the TTL-formatted ontology and do not include any explanations.
                Be as succinct as possible and only include the most relevant information.
                However, make sure to list just one concept per subject, predicate or object.  If a subject, predicate or object contains multiple concepts, split them into separate triples.
                """},
                {"role": "user", "content": text},
            ]

            outputs = pipeline(messages, max_new_tokens=2048)
            return outputs[0]["generated_text"][-1]["content"]

        else:
            text = open(f"{prefix}.json", "r").read()
            messages = [
                {"role": "system", "content": f"""You are an expert AI system that specializes in named entity recognition and knowledge graph extraction. 
                You will be given a knowledge graph in JSON format composed by RDF triples (subject, predicate, object).
                The input will conform to this JSON schema : {Output.model_json_schema()}.
                This knowledge graph was created by merging the JSON-formatted knowledge graphs of multiple input texts.
                Therefore, it may contain duplicate entitities under a different name or URI, but with the same meaning.
                It is your task to analyse the knowledge graph, merge similar or identical entities, simplify the knowledge graph as much as possible, and output an ontology as a new knowledge graph in JSON format.
                The output should conform to this JSON schema : {Output.model_json_schema()}.  
                Only output the JSON-formatted ontology and do not include any explanations.
                Be as succinct as possible and only include the most relevant information.
                However, make sure to list just one concept per subject, predicate or object.  If a subject, predicate or object contains multiple concepts, split them into separate triples.
                """},
                {"role": "user", "content": text},
            ]

            outputs = pipeline(messages, max_new_tokens=2048)
            return outputs[0]["generated_text"][-1]["content"]

    except Exception as e:
        return f"exception: {e}"


def rq2_ontology(input_file: str):
    print(f"processing input {input_file}")
    is_json = input_file.endswith(".json")
    prefix = input_file[:-5] if is_json else input_file[:-4]
    output = generate_ontology(prefix, is_json)
    with open(f"{prefix}_ontology.{'json' if is_json else 'ttl'}", "w") as f:
        f.write(output)

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : rq2.py <input-file.[json|ttl]>")
    rq2_ontology(sys.argv[1])
