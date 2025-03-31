from models import *
from typing import Callable
import json
import rdflib
import pydot

import sys



def combine_jsons(result: ModelResult, predicate: Callable[[RowResult], bool]) -> Output:
    triples = []
    for row in result.row_results:
        if predicate(row):
            cleaned = remove_markdown(row.result_json)
            try:
                data = json.loads(cleaned)
                if "triples" in data:
                    for triple in data["triples"]:
                        try:
                            row_result = Triple.model_validate_json(json.dumps(triple))
                        except:
                            row_result = None
                        if row_result is not None:
                            triples.append(row_result)
                        else:
                            print(f"skipping invalid triple for {row.url}")
                else:
                    print(f"no triples found for {row.url}")
            except Exception as e:
                print(f"could not parse json for {row.url}: {e}")
    return Output(triples=sorted(triples, key=triple_comparator))

def combine_rdfs(result: ModelResult, predicate: Callable[[RowResult], bool]) -> rdflib.Graph:
    graph = rdflib.Graph()
    triples = []
    for row in result.row_results:
        if predicate(row):
            cleaned = remove_markdown(row.result_ttl)
            try:
                local_graph = rdflib.Graph()
                local_graph.parse(data=cleaned, format="ttl")
                for triple in local_graph:
                    graph.add(triple)
            except Exception as e:
                print(f"could not parse ttl for {row.url}: {e}")
    return graph

def rdf_to_dot(graph: rdflib.Graph) -> str:
    dot = pydot.Dot(graph_type='digraph')
    for s, p, o in graph:
        s_node = pydot.Node(str(s))
        o_node = pydot.Node(str(o))
        dot.add_node(s_node)
        dot.add_node(o_node)
        dot.add_edge(pydot.Edge(s_node, o_node, label=str(p)))
    return dot.to_string()


def visualize_rdf(graph: rdflib.Graph, output_file: str):
    dot_str = rdf_to_dot(graph)
    with open(output_file, 'w') as f:
        f.write(dot_str)
    (graph,) = pydot.graph_from_dot_file(output_file)
    graph.set_layout('circo')
    graph.write_svg(output_file)

def rq2_results(input_file: str):
    print(f"processing input {input_file}")

    with open(input_file, "r") as f:
        content = f.read()
        model_result = ModelResult.model_validate_json(content)

        if model_result is None:
            raise ValueError(f"Could not parse input file {input_file}")

        # te bekijken :
        # - gecombineerde ttl / json voor alles met disinfomation
        # - gecombineerde ttl / json voor alles zonder disinfomation
        # - gecombineerde ttl voor alle documenten (hier zijn we dan wel de metadata kwijt)

        disinformation_combined_ttl = f"{sanitize_filename(model_result.model_input.model_name)}_disinformation_combined.ttl"
        disinformation_combined_ttl_svg = f"{sanitize_filename(model_result.model_input.model_name)}_disinformation_combined.svg"
        disinformation_combined_json = f"{sanitize_filename(model_result.model_input.model_name)}_disinformation_combined.json"
        no_disinformation_combined_ttl = f"{sanitize_filename(model_result.model_input.model_name)}_no_disinformation_combined.ttl"
        no_disinformation_combined_ttl_svg = f"{sanitize_filename(model_result.model_input.model_name)}_no_disinformation_combined.svg"
        no_disinformation_combined_json = f"{sanitize_filename(model_result.model_input.model_name)}_no_disinformation_combined.json"
        all_combined_ttl = f"{sanitize_filename(model_result.model_input.model_name)}_all_combined.ttl"
        all_combined_ttl_svg = f"{sanitize_filename(model_result.model_input.model_name)}_all_combined.svg"
        all_combined_json = f"{sanitize_filename(model_result.model_input.model_name)}_all_combined.json"

        print(f"processing jsons")

        with open(disinformation_combined_json, "w") as f:
            f.write(combine_jsons(model_result, lambda x: x.y == 1).model_dump_json(indent=2))
        with open(no_disinformation_combined_json, "w") as f:
            f.write(combine_jsons(model_result, lambda x: x.y == 0).model_dump_json(indent=2))
        with open(all_combined_json, "w") as f:
            f.write(combine_jsons(model_result, lambda x: True).model_dump_json(indent=2))

        print(f"processing rdfs")
        with open(disinformation_combined_ttl, "w") as f:
            g = combine_rdfs(model_result, lambda x: x.y == 1 and 'reddit' not in x.url)
            f.write(g.serialize(format="ttl"))
            visualize_rdf(g, disinformation_combined_ttl_svg)
        with open(no_disinformation_combined_ttl, "w") as f:
            g = combine_rdfs(model_result, lambda x: x.y == 0 and 'reddit' not in x.url)
            f.write(g.serialize(format="ttl"))
            visualize_rdf(g, no_disinformation_combined_ttl_svg)
        with open(all_combined_ttl, "w") as f:
            g = combine_rdfs(model_result, lambda x: 'reddit' not in x.url)
            f.write(g.serialize(format="ttl"))
            visualize_rdf(g, all_combined_ttl_svg)

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : rq2_results.py <combined-db.sqlite>")
    rq2_results(sys.argv[1])
