from pathlib import Path

import rdflib
from rdflib import Literal
from models import *
from rdf_utils import *
import sys

def rq2_join_ontology(input_folder: str, prefix: str):
    print(f"processing input_folder {input_folder}")
    triples = []
    for input_file in Path(input_folder).rglob(f"{prefix}_*.json"):
        print(f"processing input_file {input_file}")
        with open(input_file, "r") as f:
            content = f.read()
            model_result = Output.model_validate_json(content)

            if model_result is None:
                print(f"Could not parse input file {input_file}")
                continue

            triples.extend(model_result.triples)
    print(f"total nr of triples is {len(triples)}")
    with open(f"{prefix}.json", "w") as f:
        f.write(Output(triples=triples).model_dump_json(indent=2))

    graph = rdflib.Graph()
    for triple in triples:
        graph.add((Literal(triple.subject), Literal(triple.predicate), Literal(triple.object)))

    with open(f"{prefix}.ttl", "w") as f:
        f.write(graph.serialize(format="ttl"))

    # visualize_rdf(graph, f"meta-llama_Llama-3.3-70B-Instruct_disinformation_combined_ontology.svg", "twopi")
    visualize_rdf(graph, f"{prefix}.svg", "fdp")
    with open(f"{prefix}.dot", "w") as f:
        f.write(rdf_to_dot(graph))

    print(f"done")



if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : rq2.py <input-folder>")
    rq2_join_ontology(sys.argv[1], "meta-llama_Llama-3.3-70B-Instruct_disinformation_combined_ontology")
    rq2_join_ontology(sys.argv[1], "microsoft_phi-4_disinformation_combined_ontology")
