import rdflib
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Load your RDF graph
g = rdflib.Graph()
g.parse("/Users/alexander/ou/AF-tekst/images/kg.ttl", format="turtle")

# Create a directed graph
graph = nx.DiGraph()

# Add edges for certain predicates
for s, p, o in g:
    graph.add_node(str(s))
    graph.add_node(str(o))
    graph.add_edge(str(s), str(o), label=str(p))

# Use spring layout with more space
pos = nx.spring_layout(graph, k=1.0, iterations=300, seed=42)

# Plot setup
fig, ax = plt.subplots(figsize=(20, 14))

# Draw edges
nx.draw_networkx_edges(graph, pos, edge_color="gray", arrows=True, ax=ax)

# Draw edge labels
edge_labels = nx.get_edge_attributes(graph, "label")
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=28, ax=ax)

# Format node labels
node_labels = {
    n: (n.split("#")[-1] if "#" in n else n.split("/")[-1]) for n in graph.nodes
}

# Offset label positions to reduce collisions
label_offset = 0.03
for node, (x, y) in pos.items():
    ax.text(
        x + label_offset,
        y + label_offset,
        node_labels[node],
        fontsize=32,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
    )

# Adjust plot limits
x_vals, y_vals = zip(*pos.values())
x_pad, y_pad = 0.3, 0.3
ax.set_xlim(min(x_vals) - x_pad, max(x_vals) + x_pad)
ax.set_ylim(min(y_vals) - y_pad, max(y_vals) + y_pad)
ax.axis("off")
plt.tight_layout()

# Save as SVG
plt.savefig("kg.svg", format="svg", bbox_inches="tight")
plt.close()
