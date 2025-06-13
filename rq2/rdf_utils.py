import rdflib
import pydot


def rdf_to_dot(graph: rdflib.Graph) -> str:
    dot = pydot.Dot(
        graph_type="digraph",
        rankdir="LR",  # Optional: makes the graph flow left to right
        fontsize="28",
        nodesep="0.1",  # tight spacing between nodes
        ranksep="0.1",  # tight spacing between layers/ranks
        concentrate=True,  # allow edge merging if possible
    )

    added_nodes = set()

    for s, p, o in graph:
        if str(s).strip() == "" or str(o).strip() == "" or str(p).strip() == "":
            continue

        if str(s) not in added_nodes:
            dot.add_node(
                pydot.Node(
                    str(s),
                    label=str(s),
                    fontname="Helvetica-Bold",
                    fontsize="28",
                    shape="ellipse",
                )
            )
            added_nodes.add(str(s))

        if str(o) not in added_nodes:
            dot.add_node(
                pydot.Node(
                    str(o),
                    label=str(o),
                    fontname="Helvetica-Bold",
                    fontsize="28",
                    shape="ellipse",
                )
            )
            added_nodes.add(str(o))

        dot.add_edge(
            pydot.Edge(
                str(s),
                str(o),
                label=str(p),
                fontname="Helvetica",
                fontsize="22",
                arrowsize=1.2,
            )
        )

    return dot.to_string()


def visualize_rdf(
    graph: rdflib.Graph, output_file: str, layout: str = "dot", compact: bool = False
):
    dot_str = rdf_to_dot(graph)
    with open(output_file, "w") as f:
        f.write(dot_str)

    (pydot_graph,) = pydot.graph_from_dot_file(output_file)

    # Enforce tighter size and spacing
    pydot_graph.set_dpi(600)
    pydot_graph.set_size("2,2!")  # Very tight, can increase if clipping
    pydot_graph.set_graph_defaults(ranksep="0.1", nodesep="0.1")

    pydot_graph.set_layout(layout)

    if output_file.endswith(".svg"):
        pydot_graph.write_svg(output_file)
    elif output_file.endswith(".png"):
        pydot_graph.write_png(output_file)
