import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def initialize_sirs(G: nx.Graph, initial_infected=None, p_infected: float = 0.0, seed: int | None = None):
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
    seed: int | None = None,
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


def animate_sirs(G, states_history, title="SIRS Animation", interval=800, save_path=None):
    """
    Animate an SIRS (Susceptible-Infected-Recovered-Susceptible) simulation.

    Args:
        G (nx.Graph): Graph whose nodes have "state" values in S/I/R.
        states_history: List of dicts representing node->state at each time step.
                        (Output from run_sirs(..., record_node_states=True))
        title (str): Title of animation window.
        interval (int): Delay (ms) between frames.
        save_path (str): If provided, saves animation (mp4 or gif).
    """

    # Fixed layout to prevent jitter
    pos = nx.spring_layout(G, seed=42)

    # Figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color mapping for S/I/R
    STATE_COLORS = {
        "S": "#d3d3d3",   # light gray
        "I": "red",        # infected
        "R": "green"       # recovered
    }

    # --- Frame update function ---
    def update(frame):
        ax.clear()
        ax.set_title(f"{title}\nStep {frame}")

        # Current state snapshot
        state = states_history[frame]

        # Separate nodes by state
        S_nodes = [n for n in G if state[n] == "S"]
        I_nodes = [n for n in G if state[n] == "I"]
        R_nodes = [n for n in G if state[n] == "R"]

        # Draw susceptible first (background)
        if S_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=S_nodes,
                                   node_color=STATE_COLORS["S"],
                                   node_size=50, ax=ax)

        # Draw recovered next
        if R_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=R_nodes,
                                   node_color=STATE_COLORS["R"],
                                   node_size=70, ax=ax)

        # Draw infected last (on top)
        if I_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=I_nodes,
                                   node_color=STATE_COLORS["I"],
                                   node_size=100,
                                   edgecolors="black",
                                   linewidths=0.7,
                                   ax=ax)

        # Draw edges last
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)

        # Add legend
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=STATE_COLORS["S"],
                   markersize=8, label='S (Susceptible)'),

            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=STATE_COLORS["I"],
                   markeredgecolor='black',
                   markersize=8, label='I (Infected)'),

            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=STATE_COLORS["R"],
                   markersize=8, label='R (Recovered)')
        ]

        ax.legend(handles=legend_handles, loc='lower right', fontsize=8)

    # Build animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(states_history),
        interval=interval,
        repeat=False
    )

    # Save or show
    if save_path:
        ani.save(save_path, writer="ffmpeg", dpi=150)
        ani.save(save_path, writer="pillow", dpi=150)
        print(f"Saved animation to {save_path}")

    else:

        plt.show()

    return ani