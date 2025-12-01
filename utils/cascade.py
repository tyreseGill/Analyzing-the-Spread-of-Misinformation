import networkx as nx
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def get_adopters(G: nx.Graph) -> list:
    """Return all nodes whose adopter_round is finite and non-negative (excludes blocked nodes)."""
    return [ n for n, data in G.nodes(data=True)
            if 0 <= data.get("adopter_round", float("inf")) < float("inf") ]

def get_adopters_in_round(G: nx.Graph, round_num: int) -> list:
    """Return nodes adopted strictly before `round_num` (excludes blocked nodes with round=-1)."""
    return [ n for n, data in G.nodes(data=True)
            if 0 <= data.get("adopter_round", float("inf")) < round_num ]


def color_priority(color: str) -> int:
    # smaller number = drawn first
    if color in ("lightgray", "#d3d3d3"):  # background
        return 0
    return 1  # important nodes last


def perform_cascade(G: nx.Graph, threshold: float, initial_adopters: list, blocked_nodes: set = None,
                    block_budget: float = 0.05, block_per_round: int = None):
    """
    Perform a cascade process on graph G using a Linear Threshold model with dynamic blocking.

    Dynamic blocking reactively blocks high-degree adopters each round (with 1-round detection delay).
    This simulates how platforms moderate content as it spreads.

    Args:
        G: The graph on which diffusion will occur.
        threshold: Adoption threshold (fraction of adopting neighbors).
        initial_adopters: Nodes that start as adopters at round 0.
        blocked_nodes: Set of initially blocked nodes (pass empty set to enable dynamic blocking).
        block_budget: Maximum fraction of nodes to block dynamically (default 0.05 = 5%).
        block_per_round: Maximum nodes to block per round (default None = no limit, use entire budget ASAP).

    Returns:
        dict: Statistics with 'total_adopters', 'adoption_rate', 'rounds', and 'dynamically_blocked'.
    """
    # Track if dynamic blocking was requested (blocked_nodes provided, even if empty)
    enable_dynamic_blocking = blocked_nodes is not None

    if blocked_nodes is None:
        blocked_nodes = set()
    else:
        blocked_nodes = set(blocked_nodes)  # Make a copy to allow modification

    # Initialize graph with default adopter_round = inf
    for node in G.nodes():
        G.nodes[node]["adopter_round"] = float("inf")
        G.nodes[node]["p"] = 0.0
        G.nodes[node]["blocked"] = node in blocked_nodes
        G.nodes[node]["blocked_round"] = -1  # -1 means not blocked yet

    # Mark blocked nodes with special round value
    for node in blocked_nodes:
        G.nodes[node]["adopter_round"] = -1  # -1 indicates blocked

    # Mark initial adopters (skip if blocked)
    for node in initial_adopters:
        if node not in blocked_nodes:
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

        previous_adopters = set(get_adopters_in_round(G, current_round))
        print(f"\n=========== Round {current_round - 1} ===========")
        print(f"Previous adopters ({len(previous_adopters)}): {list(previous_adopters)[:10]}{'...' if len(previous_adopters) > 10 else ''}\n")

        # Dynamic blocking: block nodes that adopted in PREVIOUS round (1-round delay)
        # Only activates when blocking was explicitly requested
        if enable_dynamic_blocking and current_round > 1:
            max_blocked = int(len(G.nodes()) * block_budget)

            if len(blocked_nodes) < max_blocked:
                # Get nodes that adopted last round (detection delay = 1)
                last_round_adopters = [
                    n for n in G.nodes()
                    if G.nodes[n]["adopter_round"] == current_round - 1
                    and n not in blocked_nodes
                    and n not in initial_adopters  # Don't block seed node
                ]

                # Sort by degree (block highest-degree spreaders first)
                last_round_adopters.sort(key=lambda n: G.degree(n), reverse=True)

                # Block up to remaining budget (and per-round cap if set)
                slots = max_blocked - len(blocked_nodes)
                if block_per_round is not None:
                    slots = min(slots, block_per_round)
                to_block = last_round_adopters[:slots]

                for node in to_block:
                    blocked_nodes.add(node)
                    G.nodes[node]["adopter_round"] = -1
                    G.nodes[node]["blocked"] = True
                    G.nodes[node]["blocked_round"] = current_round  # Track when blocked

                if to_block:
                    print(f"  [Dynamic] Blocked {len(to_block)} high-degree adopters from round {current_round-1}")

        changes_this_round = 0

        # Iterate through nodes not yet adopted
        for node in G.nodes():
            # Skip if already adopted or blocked
            if node in previous_adopters or node in blocked_nodes:
                continue

            neighbors = list(G.neighbors(node))

            # Skip isolated nodes (avoid division by zero)
            if not neighbors:
                continue

            # Blocked nodes count as neighbors but can't be activated
            # This means they dilute influence but can't spread misinformation
            activated_neighbors = sum(1 for nbr in neighbors if nbr in previous_adopters and nbr not in blocked_nodes)
            p = activated_neighbors / len(neighbors)

            exceeded = p > threshold
            if exceeded:
                G.nodes[node]["adopter_round"] = current_round
                G.nodes[node]["p"] = p
                changes_this_round += 1

        if changes_this_round == 0:
            print(f"No new adopters in round {current_round}. Cascade stalled.")
            break

        current_round += 1

    final_adopters = get_adopters_in_round(G, current_round)
    print(f"\n=========== Final Round {current_round - 1} Adopters ({len(final_adopters)}) ===========")
    print(f"Total adopters: {len(final_adopters)}")

    return {
        'total_adopters': len(final_adopters),
        'adoption_rate': len(final_adopters) / total_nodes,
        'rounds': current_round - 1,
        'dynamically_blocked': len(blocked_nodes)
    }


def animate_cascade(G, title="Cascade Animation", interval=800, save_path=None, blocked_nodes=None):
    """
    Animate a cascade process over time based on each node's "adopter_round".

    Args:
        G: Graph containing "adopter_round" for each node.
        title: Title for the animation window.
        interval: Delay (ms) between frames.
        save_path: If provided, saves animation to file (e.g., "cascade.mp4").
        blocked_nodes: Set of blocked node IDs to display as squares.
    """
    if blocked_nodes is None:
        blocked_nodes = set()

    # Extract adoption rounds (exclude blocked nodes which have adopter_round = -1)
    adopter_rounds = np.array([
        data.get("adopter_round", np.inf)
        for node, data in G.nodes(data=True)
        if node not in blocked_nodes
    ])

    finite_rounds = adopter_rounds[(adopter_rounds < np.inf) & (adopter_rounds >= 0)]
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

    # Update function for animation frames
    def update(frame):
        ax.clear()

        # Separate nodes into categories: blocked (by this frame), adopted, not_adopted
        blocked = []
        adopted = []
        not_adopted = []

        for node in G.nodes():
            br = G.nodes[node].get("blocked_round", -1)
            ar = G.nodes[node].get("adopter_round", np.inf)

            # Node is blocked if blocked_round > 0 and <= current frame
            if br > 0 and br <= frame:
                blocked.append(node)
            elif 0 <= ar <= frame:
                adopted.append(node)
            else:
                not_adopted.append(node)

        blocked_label = f" (Blocked {len(blocked)} nodes)" if blocked else ""
        ax.set_title(f"{title}{blocked_label}\nCascade Round: {frame}")

        # Draw background (non-adopters) FIRST
        if not_adopted:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=not_adopted,
                node_color="#d3d3d3",
                node_size=50,
                ax=ax,
            )

        # Draw blocked nodes as SQUARES in dark gray
        if blocked:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=blocked,
                node_color="#808080",  # darker gray
                node_size=70,
                node_shape="s",  # square
                edgecolors="black",
                linewidths=1.0,
                ax=ax,
            )

        # Draw adopted nodes LAST (on top)
        if adopted:
            adopted_colors = []
            for node in adopted:
                ar = G.nodes[node]["adopter_round"]
                adopted_colors.append(cmap(ar / (max_round + 1)))

            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=adopted,
                node_color=adopted_colors,
                node_size=80,
                edgecolors="black",
                linewidths=0.8,
                ax=ax
            )

        # Draw edges last
        nx.draw_networkx_edges(
            G, pos, alpha=0.3, width=0.5, ax=ax
        )

        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#d3d3d3', markersize=8, label='Non-adopter'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#808080', markeredgecolor='black', markersize=8, label='Blocked'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.5), markeredgecolor='black', markersize=8, label='Adopter'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=max_round + 2,
        interval=interval,
        repeat=False,
    )

    if save_path:
        ani.save(save_path, writer="pillow", dpi=150)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()

    return ani
