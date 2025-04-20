import rdflib
import pydot

def rdf_to_dot(graph: rdflib.Graph) -> str:
    dot = pydot.Dot(graph_type='digraph')
    added_nodes = set()
    for s, p, o in graph:
        if str(s).strip() == '' or str(o).strip() == '' or str(p).strip() == '':
            print(f"Skipping invalid data: '{s}' -> '{p}' -> '{o}'")
            continue

        if str(s) not in added_nodes:
            s_node = pydot.Node(str(s), label=f'<<font face="boldfontname">{str(s)}</font>>', fontsize=72, style="bold")
            dot.add_node(s_node)
            added_nodes.add(str(s))

        if str(o) not in added_nodes:
            o_node = pydot.Node(str(o), label=f'<<font face="boldfontname">{str(o)}</font>>', fontsize=72, style="bold")
            dot.add_node(o_node)
            added_nodes.add(str(o))

        edge = pydot.Edge(str(s), str(o), fontsize=72, style="bold", label=f'<<font face="boldfontname">{str(p)}</font>>', arrowsize=3.0)
        dot.add_edge(edge)
    print(dot.to_string())
    return dot.to_string()


def visualize_rdf(graph: rdflib.Graph, output_file: str, layout: str = 'circo', compact: bool = False):
    dot_str = rdf_to_dot(graph)
    with open(output_file, 'w') as f:
        f.write(dot_str)
    (graph,) = pydot.graph_from_dot_file(output_file)
    if compact:
        graph.set_dpi(900)
        graph.set_size("3,3!")
    graph.set_layout(layout)
    if (output_file.endswith(".svg")):
        graph.write_svg(output_file)
    elif (output_file.endswith(".png")):
        graph.write_png(output_file)

