import numpy as np
import networkx as nx


def compute_misinformation_risk(G: nx.Graph) -> dict:
    """
    Returns the calculated risk score associated with each node in the provided graph.

    Args:
        G: The graph whose nodes are to be assessed for risk.
    """
    # PageRank
    pr = nx.pagerank(G)
    pr_vals = list(pr.values())
    pr_threshold = np.percentile(pr_vals, 90) # Top 10%

    # K-core
    num_cores = nx.core_number(G)
    max_core = max(num_cores.values())

    # Articulation Points
    articulation = set(nx.articulation_points(G))

    # Bridges (edge-level)
    bridge_counts = count_node_bridge_connections(G)

    # Clustering
    clust_coef = nx.clustering(G)
    clust_vals = list(clust_coef.values())
    clust_threshold = np.percentile(clust_vals, 75) # Top 25%

    # Betweenness
    btw = nx.betweenness_centrality(G, k=min(200, G.number_of_nodes()))
    btw_vals = list(btw.values())
    btw_threshold = np.percentile(btw_vals, 90)   # Top 10%

    risk_scores = {}

    # Tallies up risk for each node
    for n in G.nodes():
        score = 0

        # Echo chamber zone
        if clust_coef[n] >= clust_threshold:
            score += 1

        # Influential node
        if pr[n] >= pr_threshold:
            score += 2

        # Gatekeeper node
        if btw[n] >= btw_threshold:
            score += 2
        
        # Structural bottleneck
        if n in articulation:
            score += 2

        # Community-hopping nodes
        if bridge_counts[n] >= 3:
            score += 2
        elif bridge_counts[n] >= 1:
            score += 1

        # Super-spreader zone
        if num_cores[n] == max_core:
            score += 3
        
        risk_scores[n] = score

    return risk_scores


def count_node_bridge_connections(G: nx.Graph) -> dict:
    """
    Tallies up the number of bridges that each node is involved in.

    Args:
        G: The graph to be analyzed for bridges.
    """
    bridges = set(nx.bridges(G))
    bridge_counts = {
        n: 0 for n in G.nodes()
    }
    for u, v in bridges:
        bridge_counts[u] += 1
        bridge_counts[v] += 1
    return bridge_counts


def build_risk_summary(G: nx.Graph, risk_scores: dict):
    """
    Develops a structured misinformation vulnerability summary for an undirected graph.

    Args:
        G: The undirected graph to analyze.
        risk_scores: The node-risk scores from compute_misinformation_risk().
    """
    
    # 1. Graph Basics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = sum(
        dict(G.degree()).values()
    ) / num_nodes
    avg_clust = nx.average_clustering(G)

    # Gets size of largest connected component
    lcc = len(max(nx.connected_components(G), key=len))

    # 2. Core-periphery
    num_cores = nx.core_number(G)
    max_core = max(num_cores.values())
    max_core_nodes = [ n for n, k in num_cores.items() if k == max_core ]

    # 3. Articulation points & bridges
    articulation = set(nx.articulation_points(G))
    bridges = set(nx.bridges(G))

    # Count bridge connections/node
    bridge_counts = count_node_bridge_connections(G)

    # 4. Influence Metrics
    pr = nx.pagerank(G)
    deg = dict(G.degree())
    btw = nx.betweenness_centrality(G, k=min(200, num_nodes))

    def top(d: dict, k: int=10) -> list:
        """
        Returns k number of values from key-value pairs based on highest value.

        Args:
            d: The dictionary from which the top-ranked pairs will be pulled from.
            k: The number of top-ranked pairs that will be pulled.
        """
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]

    # 5. Risk score analysis
    risk_vals = list(risk_scores.values())
    avg_risk = np.mean(risk_vals)
    max_risk = max(risk_vals)
    high_risk_nodes = sorted(
        [ node for node, score in risk_scores.items() if score >= max_risk - 1 ],
        key=lambda n: risk_scores[n],
        reverse=True
    )

    # 6. Echo chambers
    clustering = nx.clustering(G)
    top_clust = top(clustering)

    # 7. Nodes touching many bridges
    top_bridge_nodes = top(bridge_counts)

    # Build summary
    summary = {
        "Graph Overview": {
            "nodes": num_nodes,
            "edges": num_edges,
            "average_degree": round(avg_degree, 3),
            "average_clustering": round(avg_clust, 3),
            "largest_component_size": lcc
        },
        "Core-Periphery": {
            "max_k_core": max_core,
            "num_core_nodes": len(max_core_nodes),
            "top_core_nodes": max_core_nodes[:10]
        },
        "Structural Weakpoints": {
            "num_articulation_points": len(articulation),
            "num_bridges": len(bridges),
            "top_bridge_nodes": top_bridge_nodes
        },
        "Influence": {
            "top_pagerank": top(pr),
            "top_betweeness": top(btw),
            "top_degree": top(deg)
        },
        "Risk Analysis": {
            "average_risk": round(avg_risk, 3),
            "max_risk": max_risk,
            "top_risk_nodes": high_risk_nodes[:10]
        },
        "Echo Chambers": {
            "top_clustering_nodes": top_clust
        }
    }

    return summary


def print_summary(summary: dict):
    """
    Outputs summary to console in organized format.

    Args:
        summary: The dictionary of text to be displayed
    """
    for section, items in summary.items():
        print(f"\n=========== {section} ===========")
        for key, val in items.items():
            print(f"{key}: {val}")

