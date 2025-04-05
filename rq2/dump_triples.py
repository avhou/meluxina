from models import *
import os
import sys

def dump_triples(triple_file: str):
    with open(triple_file, "r") as f:
        triple_generation_model = os.path.splitext(os.path.basename(triple_file))[0]
        content = f.read()
        triple_model = Output.model_validate_json(content)
        print("\n".join([f"{t}" for t in triple_model.triples]))

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : dump_triples.py <input_file>")
    dump_triples(sys.argv[1])
