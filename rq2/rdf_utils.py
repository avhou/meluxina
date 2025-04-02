import rdflib
import pydot

def rdf_to_dot(graph: rdflib.Graph) -> str:
    dot = pydot.Dot(graph_type='digraph')
    for s, p, o in graph:
        s_node = pydot.Node(str(s))
        o_node = pydot.Node(str(o))
        dot.add_node(s_node)
        dot.add_node(o_node)
        dot.add_edge(pydot.Edge(s_node, o_node, label=str(p)))
    return dot.to_string()


def visualize_rdf(graph: rdflib.Graph, output_file: str, layout: str = 'circo'):
    dot_str = rdf_to_dot(graph)
    with open(output_file, 'w') as f:
        f.write(dot_str)
    (graph,) = pydot.graph_from_dot_file(output_file)
    graph.set_layout(layout)
    graph.write_svg(output_file)

