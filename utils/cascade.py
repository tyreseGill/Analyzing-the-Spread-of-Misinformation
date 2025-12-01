import networkx as nx
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

def get_adopters(G: nx.Graph) -> list:
    """Return all nodes whose adopter_round is finite."""
    return [ n for n, data in G.nodes(data=True)
            if data.get("adopter_round", float("inf")) < float("inf") ]

def get_adopters_in_round(G: nx.Graph, round_num: int) -> list:
    """Return nodes adopted strictly before `round_num`."""
    return [ n for n, data in G.nodes(data=True)
            if data.get("adopter_round", float("inf")) < round_num ]


def color_priority(color: str) -> int:
    # smaller number = drawn first
    if color in ("lightgray", "#d3d3d3"):  # background
        return 0
    return 1  # important nodes last


def perform_cascade(G: nx.Graph, threshold: float, initial_adopters: list):
    """
    Perform a cascade process on graph G using a simple Linear Threshold model.

    Args:
        G: The graph on which diffusion will occur.
        threshold: Adoption threshold (fraction of adopting neighbors).
        initial_adopters: Nodes that start as adopters at round 0.
    """
    
    # Initialize graph with default adopter_round = inf
    for node in G.nodes():
        G.nodes[node].setdefault("adopter_round", float("inf"))
        G.nodes[node].setdefault("p", 0.0)

    # Mark initial adopters
    for node in initial_adopters:
        G.nodes[node]["adopter_round"] = 0

    total_nodes = len(G)
    current_round = 1

    # Continue until adoption proportion meets threshold
    while True:
        adopters = get_adopters(G)
        adoption_rate = len(adopters) / total_nodes

        if adoption_rate >= threshold:
            print(f"Stopping cascade: adoption rate {round(adoption_rate, 3)} >= threshold {threshold}.")
            break
        
        previous_adopters = get_adopters_in_round(G, current_round)
        print(f"\n=========== Round {current_round - 1} ===========")
        print(f"Previous adopters ({len(previous_adopters)}): {previous_adopters}\n")

        changes_this_round = 0

        # Iterate through nodes not yet adopted
        for node in G.nodes():
            if node in previous_adopters:
                continue

            neighbors = list(G.neighbors(node))

            # Skip isolated nodes (avoid division by zero)
            if not neighbors:
                continue

            activated_neighbors = sum(1 for nbr in neighbors if nbr in previous_adopters)
            p = activated_neighbors / len(neighbors)

            exceeded = p > threshold
            if exceeded:
                G.nodes[node]["adopter_round"] = current_round
                G.nodes[node]["p"] = p
                changes_this_round += 1

            # print(f"{node}: ({activated_neighbors}/{len(neighbors)}) p={round(p,3)}"
                #   f" > {threshold}  -> {'TRUE' if exceeded else 'FALSE'}")

        if changes_this_round == 0:
            print(f"No new adopters in round {current_round}. Cascade stalled.")
            break

        current_round += 1

    final_adopters = get_adopters_in_round(G, current_round)
    print(f"\n=========== Final Round {current_round - 1} Adopters ({len(final_adopters)}) ===========")
    print(final_adopters)


def animate_cascade(G, title="Cascade Animation", interval=800, save_path=None):
    """
    Animate a cascade process over time based on each node’s "adopter_round".

    Args:
        G: Graph containing "adopter_round" for each node.
        title: Title for the animation window.
        interval: Delay (ms) between frames.
        save_path: If provided, saves animation to file (e.g., "cascade.mp4").
    """
    # Extract adoption rounds
    adopter_rounds = np.array([
        data.get("adopter_round", np.inf)
        for _, data in G.nodes(data=True)
    ])

    finite_rounds = adopter_rounds[adopter_rounds < np.inf]
    if len(finite_rounds) == 0:
        print("No cascade occurred — no adopters found.")
        return

    max_round = int(finite_rounds.max())

    # Fixed layout to prevent jitter
    pos = nx.spring_layout(G, seed=42)

    # Figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color map
    cmap = plt.cm.get_cmap("OrRd")

    # Precompute a node color list for each frame
    frame_colors = []
    for r in range(max_round + 2):  # include "final" frame
        colors = []
        for node in G.nodes():
            ar = G.nodes[node].get("adopter_round", np.inf)

            if ar == np.inf:
                colors.append("#d3d3d3")           # never adopts
            elif ar <= r:
                colors.append(cmap(ar / (max_round + 1)))  # shade by adoption timing
            else:
                colors.append("#d3d3d3")           # not yet adopted at this frame
        frame_colors.append(colors)

    # Update function for animation frames
    def update(frame):
        ax.clear()
        ax.set_title(f"{title}\nCascade Round: {frame}")

        # Separate nodes into adopted vs not adopted
        adopted = []
        not_adopted = []
        
        for _, node in enumerate(G.nodes()):
            ar = G.nodes[node].get("adopter_round", np.inf)
            if ar <= frame:
                adopted.append(node)
            else:
                not_adopted.append(node)

        # Draw background (non-adopters) FIRST
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=not_adopted,
            node_color="#d3d3d3",
            node_size=50,
            ax=ax,
        )

        # Draw adopted nodes SECOND (on top)
        adopted_colors = []
        for node in adopted:
            ar = G.nodes[node]["adopter_round"]
            adopted_colors.append(cmap(ar / (max_round + 1)))

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=adopted,
            node_color=adopted_colors,
            node_size=80,          # larger so it visually stands above
            edgecolors="black",
            linewidths=0.8,
            ax=ax
        )

        # Draw edges last
        nx.draw_networkx_edges(
            G, pos, alpha=0.3, width=0.5, ax=ax
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=max_round + 2,
        interval=interval,
        repeat=False,
    )

    if save_path:
        ani.save(save_path, writer="ffmpeg", dpi=150)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()

    return ani
