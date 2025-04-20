from rdf_utils import *
import rdflib
import sys


def visualize(input_file: str, output_file: str, layout: str):
    print(f"processing input {input_file}")

    with open(output_file, "w") as output:
        g = rdflib.Graph()
        g.parse(input_file, format="turtle")
        visualize_rdf(g, output_file, layout, compact=True)

if __name__ == "__main__":
    if len(sys.argv) <= 2:
        raise RuntimeError("usage : visualize_rdf.py <input.ttl> <output.png> {layout}")
    visualize(sys.argv[1], sys.argv[2], 'circo' if len(sys.argv) <= 2 else sys.argv[3])
