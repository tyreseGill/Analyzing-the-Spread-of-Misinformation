import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.animation as animation
import numpy as np
from matplotlib.lines import Line2D

POLITICIAN_RELATIONSHIPS = "data/fb-pages-politician/fb-pages-politician.edges"

G = nx.Graph()

with open(POLITICIAN_RELATIONSHIPS) as file:
    for line in file:
        u, v = line.split(",")
        u = int(u)
        v = int(v)
        G.add_edge(u, v)

# built-in community detection
communities = list(greedy_modularity_communities(G))
print(f"Found {len(communities)} communities")

node_community_mapping = {}
for i, community in enumerate(communities):
    for node in community:
        node_community_mapping[node] = i

# Count edges between communities
between_edges = {}

for u, v in G.edges():
    cu = node_community_mapping[u]
    cv = node_community_mapping[v]
    if cu != cv:
        key = tuple(sorted((cu, cv)))
        if key not in between_edges:
            between_edges[key] = 1
        else:
            between_edges[key] += 1

# select sizable community with small amount of weak ties
min_community_size = 50
best_pair = None
best_score = float("inf")

for (c1, c2), between_edge_count in between_edges.items():
    size1 = len(communities[c1])
    size2 = len(communities[c2])
    # skip over communities not meeting threshold
    if size1 < min_community_size or size2 < min_community_size:
        continue
    score = between_edge_count / (size1 + size2)

    if score < best_score:
        best_score = score
        best_pair = (c1, c2)

if best_pair is None:
    print("no pair found large enough")
    raise ValueError()

c1, c2 = best_pair

community1 = communities[c1]
community2 = communities[c2]

chosen_community_nodes = set(community1) | set(community2)
community_graph = G.subgraph(chosen_community_nodes).copy()

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(community_graph, seed=42)

# color the communities
color_map = {}
for node in community1:
    if node in community_graph:
        color_map[node] = "tab:blue"

for node in community2:
    if node in community_graph:
        color_map[node] = "tab:red"

c1_edges = []
c2_edges = []
edges_between = []

community1_set = set(community1)
community2_set = set(community2)

for u, v in community_graph.edges():
    if u in community1_set and v in community1_set:
        c1_edges.append((u, v))
    elif u in community2_set and v in community2_set:
        c2_edges.append((u, v))
    else:
        edges_between.append((u, v))

nx.draw_networkx_edges(community_graph, pos, edgelist=c1_edges, alpha=0.3)
nx.draw_networkx_edges(community_graph, pos, edgelist=c2_edges, alpha=0.3)
nx.draw_networkx_edges(community_graph, pos, edgelist=edges_between, alpha=0.1, style="dashed")

nx.draw_networkx_nodes(
    community_graph,
    pos,
    nodelist=list(community1_set),
    node_color="tab:blue",
    node_size=30,
    label=f"Community {c1}",
)
nx.draw_networkx_nodes(
    community_graph,
    pos,
    nodelist=list(community2_set),
    node_color="tab:red",
    node_size=30,
    label=f"Community {c2}",
)

plt.axis("off")
plt.legend()
plt.title("Two communities with weak ties among Facebook politician pages")
plt.tight_layout()
plt.show()

def get_adopters(G: nx.Graph) -> list:
    """Return all nodes whose adopter_round is finite and non-negative (excludes blocked nodes)."""
    return [
        n
        for n, data in G.nodes(data=True)
        if 0 <= data.get("adopter_round", float("inf")) < float("inf")
    ]


def get_adopters_in_round(G: nx.Graph, round_num: int) -> list:
    """Return nodes adopted strictly before `round_num` (excludes blocked nodes with round=-1)."""
    return [
        n
        for n, data in G.nodes(data=True)
        if 0 <= data.get("adopter_round", float("inf")) < round_num
    ]


def perform_cascade(
    G: nx.Graph,
    threshold: float,
    initial_adopters: list,
    blocked_nodes: set = None,
    block_budget: float = 0.05,
    block_per_round: int = None,
):
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
        block_per_round: Maximum nodes to block per round (default None = no limit).

    Returns:
        dict: Statistics with 'total_adopters', 'adoption_rate', 'rounds', and 'dynamically_blocked'.
    """
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

    for node in blocked_nodes:
        G.nodes[node]["adopter_round"] = -1  # -1 indicates blocked

    # Mark initial adopters (skip if blocked)
    for node in initial_adopters:
        if node not in blocked_nodes:
            G.nodes[node]["adopter_round"] = 0

    total_nodes = len(G)
    current_round = 1

    while True:
        adopters = get_adopters(G)
        adoption_rate = len(adopters) / total_nodes

        if adoption_rate >= threshold:
            print(
                f"Stopping cascade: adoption rate {round(adoption_rate, 3)} >= threshold {threshold}."
            )
            break

        previous_adopters = set(get_adopters_in_round(G, current_round))
        print(f"\n=========== Round {current_round - 1} ===========")
        print(
            f"Previous adopters ({len(previous_adopters)}): "
            f"{list(previous_adopters)[:10]}{'...' if len(previous_adopters) > 10 else ''}\n"
        )

        if enable_dynamic_blocking and current_round > 1:
            max_blocked = int(len(G.nodes()) * block_budget)

            if len(blocked_nodes) < max_blocked:
                # Get nodes that adopted last round
                last_round_adopters = [
                    n
                    for n in G.nodes()
                    if G.nodes[n]["adopter_round"] == current_round - 1
                    and n not in blocked_nodes
                    and n not in initial_adopters
                ]

                # Sort by degree
                last_round_adopters.sort(key=lambda n: G.degree(n), reverse=True)

                # Block up to remaining budget
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
                    print(
                        f"  [Dynamic] Blocked {len(to_block)} high-degree adopters from round {current_round-1}"
                    )

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
            activated_neighbors = sum(
                1
                for nbr in neighbors
                if nbr in previous_adopters and nbr not in blocked_nodes
            )
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
    print(
        f"\n=========== Final Round {current_round - 1} Adopters ({len(final_adopters)}) ==========="
    )
    print(f"Total adopters: {len(final_adopters)}")

    return {
        "total_adopters": len(final_adopters),
        "adoption_rate": len(final_adopters) / total_nodes,
        "rounds": current_round - 1,
        "dynamically_blocked": len(blocked_nodes),
    }


def animate_cascade(
    G,
    title="Cascade Animation",
    interval=800,
    save_path=None,
):
    """
    Animate a cascade process over time based on each node's "adopter_round".

    Args:
        G: Graph containing "adopter_round" for each node.
        title: Title for the animation window.
        interval: Delay (ms) between frames.
        save_path: If provided, saves animation to file (e.g., "cascade.mp4").
    """
    adopter_rounds = np.array(
        [
            data.get("adopter_round", np.inf)
            for node, data in G.nodes(data=True)
            if data.get("adopter_round", np.inf) >= 0
        ]
    )

    finite_rounds = adopter_rounds[
        (adopter_rounds < np.inf) & (adopter_rounds >= 0)
    ]
    if len(finite_rounds) == 0:
        print("No cascade occurred — no adopters found.")
        return

    max_round = int(finite_rounds.max())

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.get_cmap("OrRd")

    def update(frame):
        ax.clear()

        blocked = []
        adopted = []
        not_adopted = []

        for node in G.nodes():
            br = G.nodes[node].get("blocked_round", -1)
            ar = G.nodes[node].get("adopter_round", np.inf)

            if br > 0 and br <= frame:
                blocked.append(node)
            elif 0 <= ar <= frame:
                adopted.append(node)
            else:
                not_adopted.append(node)

        blocked_label = f" (Blocked {len(blocked)} nodes)" if blocked else ""
        ax.set_title(f"{title}{blocked_label}\nCascade Round: {frame}")

        if not_adopted:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=not_adopted,
                node_color="tab:blue",
                node_size=50,
                ax=ax,
            )

        if blocked:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=blocked,
                node_color="tab:red",
                node_size=70,
                node_shape="s",
                edgecolors="black",
                linewidths=1.0,
                ax=ax,
            )

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
                ax=ax,
            )

        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#d3d3d3",
                markersize=8,
                label="Non-adopter",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="#808080",
                markeredgecolor="black",
                markersize=8,
                label="Blocked",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(0.5),
                markeredgecolor="black",
                markersize=8,
                label="Adopter",
            ),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

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


def main():
    seed_node = max(community1_set, key=lambda n: community_graph.degree(n))
    initial_adopters = [seed_node]
    print(f"Initial adopter (seed): {seed_node}")

    # Simulation #1: Without dynamic blocking
    cascade_stats = perform_cascade(
        community_graph,
        threshold=0.2,
        initial_adopters=initial_adopters
    )

    print("Cascade stats:", cascade_stats)

    animate_cascade(
        community_graph,
        title="Cascade in two politician communities",
        interval=800,
        save_path="img/cascade_political_parties.gif",
    )

    # Simulation #2: With dynamic blocking
    cascade_stats = perform_cascade(
        community_graph,
        threshold=0.2,
        initial_adopters=initial_adopters,
        blocked_nodes=set(),
        block_budget=0.05,
        block_per_round=5,
    )

    print("Cascade stats:", cascade_stats)

    animate_cascade(
        community_graph,
        title="Cascade in two politician communities",
        interval=800,
        save_path="img/dynamic_blocking_political_parties.gif",
    )


main()