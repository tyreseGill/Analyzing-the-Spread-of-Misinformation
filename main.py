import argparse
import networkx as nx
import os
import pandas as pd
import sys
from utils import file_exists, has_file_extension, file_empty, remove_trailing_digits, validate_color, visualize_graph, compute_misinformation_risk, get_most_influential_node, perform_cascade, animate_cascade, initialize_sirs, run_sirs, animate_sirs


class SafeArgumentParser(argparse.ArgumentParser):
    """
    A subclass of argparse.ArgumentParser that overrides the default error behavior.

    Args:
        argparse.ArgumentParser: Object for parsing command line strings into Python objects.

    Methods:
        error(message): Overrides the default error handler to print a custom error message
            and terminate the program with a non-zero exit code.
    """
    def error(self, message: str):
        # Override the default behavior (which prints to stderr and exits)
        print(f"Error: {message}.")
        sys.exit(2)


def parse_args() -> dict:
    """
    Manages input parameters for the command-line interface.

    Returns:
        dict: Collection of arguments and the associated input values provided by the user.
    """
    parser = SafeArgumentParser()

    # Defines expected arguments and behavior for CLI
    parser.add_argument(
        "file_path",
        type=str,
        help="The path to an input graph file (.csv, .edges, etc.)."
    )

    parser.add_argument(
        "--color",
        type=str,
        help="The coloration of the plotted nodes."
    )
    parser.add_argument(
        "--title",
        type=str,
        help="The name of the plot."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=500,
        help="The number of nodes to be plotted."
    )
    parser.add_argument(
        "--risk_assessment",
        action="store_true",
        help="Colors nodes based on risk score."
    )
    parser.add_argument(
        "--bow_tie",
        action="store_true",
        help="Colors nodes based on how they fit in bow-tie structure."
    )
    parser.add_argument(
        "--cascade",
        action="store_true"
    )
    parser.add_argument(
        "--sirs",
        action="store_true"
    )

    # Returns dictionary of the parsed arguments
    return vars(parser.parse_args())


def verify_file(input_file: str) -> bool:
    if not input_file:
        print("Error: No file path provided.")
        return False
    if not file_exists(input_file):
        return False
    if file_empty(input_file):
        return False
    return True


def read_graph(file_path: str) -> nx.Graph:
    """
    Reads a graph from .csv, .edges, .txt, .edgelist, .gml, or .graphml files.
    Automatically detects type based on extension.
    """

    if not verify_file(file_path):
        return None

    ext = os.path.splitext(file_path)[1].lower()

    try:
        # ----- CSV file (edge list stored in columns) -----
        if ext == ".csv":
            df = pd.read_csv(file_path)
            cols = df.columns
            src = cols[0]
            dst = cols[1]
            G = nx.from_pandas_edgelist(df, source=src, target=dst)
            return G

        # ----- SNAP / GEMSEC edge list (.edges) -----
        elif ext == ".edges":
            G = nx.read_edgelist(file_path, nodetype=int)
            return G

        # ----- Generic whitespace edge list -----
        elif ext in [".txt", ".edgelist"]:
            G = nx.read_edgelist(file_path)
            return G

        # ----- GraphML -----
        elif ext == ".graphml":
            return nx.read_graphml(file_path)

        # ----- GML -----
        elif ext == ".gml":
            return nx.read_gml(file_path)

        else:
            print(f"Error: Unsupported file extension '{ext}'.")
            print("Supported: .csv, .edges, .txt, .edgelist, .gml, .graphml")
            return None

    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None



def main():
    params = parse_args()

    # Extracts parsed arguments
    file_path = params["file_path"]
    color = params["color"]
    title = params["title"]
    sample_size = params["sample_size"]
    risk_assessment = params["risk_assessment"]
    bow_tie = params["bow_tie"]
    cascade = params["cascade"]
    sirs = params["sirs"]

    G = None

    # Check to ensure user-provided color is valid
    if color:
        color = validate_color(color)

    if file_path:
        G = read_graph(file_path)

        # Check to ensure .csv graph data is given
        if not G:
            print("Error: Unable to read file.")
            return

        if nx.is_directed(G):
            G = G.to_undirected()

        G.remove_edges_from(nx.selfloop_edges(G))

        if G.number_of_nodes() == 0:
            print("Error: Graph is empty.")
            return
        
        # Displays multiple communities with color-coding
        if "facebook_large" in file_path:
            H = visualize_graph(G,
                            title,
                            sample_size,
                            color_code=True,
                            risk_assessment=risk_assessment,
                            bow_tie=bow_tie)
        # Displays single community with uniform color
        else:
            H = visualize_graph(G,
                            title,
                            sample_size,
                            color=color,
                            risk_assessment=risk_assessment,
                            bow_tie=bow_tie)
            
        if cascade:
            risk_scores = compute_misinformation_risk(H)
            riskiest_node = get_most_influential_node(H, risk_scores)
            perform_cascade(H, threshold=0.1, initial_adopters=[riskiest_node])
            animate_cascade(H, title="Cascade Spread", interval=600)
        
        if sirs:
            initialize_sirs(H, p_infected=0.02, seed=42)

            history = run_sirs(
                H,
                beta=0.25,
                gamma=0.1,
                mu=0.05,
                seed=42,
                max_steps=50,
                record_node_states=True
            )

            animate_sirs(
                H,
                history["states"],
                title="SIRS Spread",
                interval=600,
                save_path="img/sirs.gif",
                risk_assessment=risk_assessment,
                bow_tie=bow_tie
            )
    else:
        print("Error: A file was not provided.")
        return


if __name__ == "__main__":
    main()
