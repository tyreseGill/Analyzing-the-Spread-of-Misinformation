import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.visualize import color_nodes_by_risk, color_nodes_by_bow_tie


def initialize_sirs(G: nx.Graph, initial_infected=None, p_infected: float = 0.0, seed: int = 42):
    """
    Initialize SIRS states on the graph.

    Each node gets a 'state' attribute in {"S", "I", "R"}.

    Args:
        G: The graph.
        initial_infected: Optional iterable of nodes to mark as initially infected.
        p_infected: If initial_infected is None, infect each node independently with this probability.
        seed: RNG seed for reproducibility.
    """
    rng = random.Random(seed)

    # Set everyone susceptible initially
    for n in G.nodes():
        G.nodes[n]["state"] = "S"

    if initial_infected is not None:
        for n in initial_infected:
            if n in G:
                G.nodes[n]["state"] = "I"
    else:
        for n in G.nodes():
            if rng.random() < p_infected:
                G.nodes[n]["state"] = "I"

def run_sirs(
    G: nx.Graph,
    beta: float,
    gamma: float,
    mu: float,
    max_steps: int = 50,
    seed: int = 42,
    record_node_states: bool = False,
):
    """
    Run a discrete-time SIRS process on graph G.

    Args:
        G: Graph with 'state' per node in {"S", "I", "R"}.
        beta: Infection rate per contact (S node exposed to I neighbors).
        gamma: Recovery rate (I -> R).
        mu: Immunity loss rate (R -> S).
        max_steps: Maximum number of time steps to simulate.
        seed: RNG seed.
        record_node_states: If True, keep full node->state history per step.
                            If False, only aggregate S/I/R counts per step.

    Returns:
        history: dict with keys:
            - "counts": list of dicts per step: {"S": int, "I": int, "R": int}
            - "states": list of dict[node -> state] per step (only if record_node_states=True)
    """
    rng = random.Random(seed)

    # Helper to read current states into a plain dict
    def get_state_dict():
        return {n: G.nodes[n].get("state", "S") for n in G.nodes()}

    # History containers
    counts_history = []
    states_history = [] if record_node_states else None

    # Initial snapshot
    current_states = get_state_dict()

    def count_SIR(states):
        vals = list(states.values())
        return {
            "S": vals.count("S"),
            "I": vals.count("I"),
            "R": vals.count("R"),
        }

    counts_history.append(count_SIR(current_states))
    if record_node_states:
        states_history.append(current_states.copy())

    # --- Time evolution ---
    for _ in range(1, max_steps + 1):
        new_states = {}

        for node in G.nodes():
            state = current_states[node]

            if state == "S":
                # Count infected neighbors
                infected_neighbors = sum(
                    1 for nbr in G.neighbors(node) if current_states[nbr] == "I"
                )

                if infected_neighbors == 0:
                    new_states[node] = "S"
                else:
                    # Probability this node becomes infected:
                    # P(infection) = 1 - (1 - beta)^(#I neighbors)
                    p_inf = 1.0 - (1.0 - beta) ** infected_neighbors
                    if rng.random() < p_inf:
                        new_states[node] = "I"
                    else:
                        new_states[node] = "S"

            elif state == "I":
                # Recover with probability gamma
                if rng.random() < gamma:
                    new_states[node] = "R"
                else:
                    new_states[node] = "I"

            elif state == "R":
                # Lose immunity with probability mu
                if rng.random() < mu:
                    new_states[node] = "S"
                else:
                    new_states[node] = "R"
            else:
                # Unknown state fallback
                new_states[node] = "S"

        # Update current state
        current_states = new_states

        # Write back into graph node attributes (optional but useful)
        for n, s in current_states.items():
            G.nodes[n]["state"] = s

        counts_history.append(count_SIR(current_states))
        if record_node_states:
            states_history.append(current_states.copy())

        # Optional: early stopping if no infected remain
        if counts_history[-1]["I"] == 0:
            break

    history = {"counts": counts_history}
    if record_node_states:
        history["states"] = states_history

    return history


def animate_sirs(
    G,
    states_history,
    title="SIRS Animation",
    interval=800,
    save_path=None,
    risk_assessment=False,
    bow_tie=False
):
    """
    Animate an SIRS process on the SAME color scheme used in visualize_sample().
    Susceptible (S), Infected (I), and Recovered (R) override the base colors.

    Args:
        G: Graph (subgraph H) with nodes and positions.
        states_history: Output from run_sirs(..., record_node_states=True)
        risk_assessment: If True, use risk-based color map as baseline.
        bow_tie: If True, use bow-tie color map as baseline.
    """

    # --- FIXED LAYOUT FOR CONSISTENCY ---
    pos = nx.spring_layout(G, k=0.95, iterations=50, seed=42)

    # --- BASE COLOR MAP (from visualization) ---
    if risk_assessment:
        base_color_map, type_to_color = color_nodes_by_risk(G)
    elif bow_tie:
        base_color_map, type_to_color = color_nodes_by_bow_tie(G)
    else:
        # default: uniform color
        base_color_map = {n: "tab:blue" for n in G.nodes()}
        type_to_color = {"Nodes": "tab:blue"}

    # --- SIRS STATE COLORS ---
    STATE_COLORS = {
        "S": None,          # uses baseline color
        "I": "red",         # infected overrides
        "R": "green"        # recovered overrides
    }

    # --- FIGURE SETUP ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- Frame update function ---
    def update(frame):
        ax.clear()
        ax.set_title(f"{title}\nStep {frame}", fontsize=12)

        # States at this time step
        state = states_history[frame]

        # Build per-node colors combining baseline + SIRS state
        frame_colors = []
        for node in G.nodes():
            s = state[node]                # S, I, or R
            if STATE_COLORS[s] is None:
                frame_colors.append(base_color_map[node])
            else:
                frame_colors.append(STATE_COLORS[s])

        # Draw the graph
        nx.draw(
            G,
            pos,
            nodelist=list(G.nodes()),
            node_color=frame_colors,
            node_size=70,
            edge_color="gray",
            width=0.3,
            with_labels=False,
            ax=ax,
        )

        # --- ADD LEGEND ---
        from matplotlib.lines import Line2D

        legend_entries = []

        # If using risk or bowtie, show that legend first
        for label, color in type_to_color.items():
            legend_entries.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color, markersize=8,
                       label=label)
            )

        # Then add SIRS legend
        legend_entries.extend([
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=STATE_COLORS["S"] or "gray",
                   markersize=8, label='S (Susceptible)'),

            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=STATE_COLORS["I"],
                   markeredgecolor="black",
                   markersize=8, label='I (Infected)'),

            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=STATE_COLORS["R"],
                   markersize=8, label='R (Recovered)')
        ])

        ax.legend(
            handles=legend_entries,
            loc="lower right",
            fontsize=8,
            frameon=True,
            title="Legend"
        )

    # --- ANIMATION OBJECT ---
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(states_history),
        interval=interval,
        repeat=False
    )

    # --- SAVING OR SHOWING ---
    if save_path:
        ext = save_path.lower().split(".")[-1]
        if ext == "gif":
            ani.save(save_path, writer="pillow", dpi=150)
        elif ext == "mp4":
            try:
                ani.save(save_path, writer="ffmpeg", dpi=150)
            except:
                print("FFmpeg not found, saving as GIF instead.")
                save_path = save_path.replace(".mp4", ".gif")
                ani.save(save_path, writer="pillow", dpi=150)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()

    return ani
