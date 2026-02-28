"""Visualization utility for neurosymbolic reasoning."""

import matplotlib.pyplot as plt
import networkx as nx
import argparse
import torch
import os
from neurosymbolic.integration import NeurosymbolicSystem

def visualize_scene_and_proofs(model, image, query=None, output_path="reasoning_viz.png"):
    """Generate a visualization of the symbolic scene and reasoning proofs."""

    # Run model
    with torch.no_grad():
        output = model.forward(image)

    # Create graph
    G = nx.DiGraph()

    # 1. Add symbolic scene nodes (from perception)
    # The output structure is output["perception"]["symbolic"] -> list of lists (one per batch)
    symbolic_scene = output["perception"]["symbolic"][0]
    for item in symbolic_scene:
        # Each item is (concept, confidence, attributes)
        concept, conf, _ = item
        G.add_node(concept, type="concept", confidence=conf, color="lightblue")
        G.add_edge("Scene", concept)

    # 2. Add derived facts
    # reasoning is a list of results, one per scene/object
    reasoning_list = output["reasoning"]
    for reasoning in reasoning_list:
        for pred, args, conf, source in reasoning["derived_facts"]:
            if source == "derived":
                fact_label = f"{pred}({','.join(args)})"
                G.add_node(fact_label, type="derived", confidence=conf, color="lightgreen")
                # In a real trace we'd link to premises, here we just show they are derived
                G.add_edge("Reasoning", fact_label)

    # 3. Add query proof if provided
    if query:
        proofs = model.query(image, query)
        if proofs:
            best_proof = proofs[0]
            # Link proof steps
            for i in range(len(best_proof["proof"]) - 1):
                G.add_edge(best_proof["proof"][i+1], best_proof["proof"][i], label="proves")

    # Plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)

    node_colors = [G.nodes[n].get("color", "gray") for n in G.nodes]

    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            node_size=2000, font_size=10, font_weight="bold", arrows=True)

    plt.title(f"NeuroSymbolic-T4 Reasoning Visualization\nQuery: {query if query else 'None'}")
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize neurosymbolic reasoning")
    parser.add_argument("--query-pred", type=str, default="dangerous")
    parser.add_argument("--query-arg", type=str, default="obj0")
    parser.add_argument("--output", type=str, default="reasoning_visualization.png")
    args = parser.parse_args()

    # Initialize model
    model = NeurosymbolicSystem()
    model.eval()

    # Dummy image
    image = torch.randn(1, 3, 224, 224)

    query = (args.query_pred, (args.query_arg,))

    visualize_scene_and_proofs(model, image, query, args.output)

if __name__ == "__main__":
    main()
