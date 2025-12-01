import argparse
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Tkinter issues
import networkx as nx
import pandas as pd
from utils import file_exists, has_file_extension, file_empty, remove_trailing_digits, validate_color, visualize_graph, compute_misinformation_risk, get_most_influential_node, perform_cascade, animate_cascade


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
        "csv_file_pathway",
        type=str,
        help='The path to the inputted .csv graph file.'
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
        action="store_true",
        help="Run Linear Threshold cascade from riskiest node."
    )
    parser.add_argument(
        "--block",
        action="store_true",
        help="Enable dynamic blocking intervention (reactively blocks high-degree adopters each round)."
    )
    parser.add_argument(
        "--block_pct",
        type=float,
        default=0.05,
        help="Fraction of nodes to block (default: 0.05 = 5%%)."
    )
    parser.add_argument(
        "--block_per_round",
        type=int,
        default=None,
        help="Max nodes to block per round (default: None = use entire budget ASAP)."
    )

    # Returns dictionary of the parsed arguments
    return vars(parser.parse_args())


def verify_compatible_file(input_file: str, file_extension: str) -> bool:
    """
    Checks to see if the file inputted is compatible with the program. If not, forces early return.

    Args:
        input_file: The name of the .csv file containing graph data.
        file_extension: The extension that a file is expected to end with.
    """
    if not input_file:
        print("Error: No file path provided.")
        return False
    if not has_file_extension(input_file, file_extension):
        return False
    if not file_exists(input_file):
        return False
    if file_empty(input_file):
        return False

    return True


def read_csv_data(file_path: str) -> nx.Graph:
    """
    Acts as gaurdrail when error occurs attempting to read file.

    Args:
        file_path: The name of the .csv file containing graph data.
    """
    # Perform early return if file is not compatible
    if not verify_compatible_file(file_path, ".csv"):
        return None

    # If program reaches this point, the .csv file provided must be valid
    try:
        df = pd.read_csv(file_path)
        column_names = df.columns
        src_tar_substr = remove_trailing_digits(column_names[0])
        G = nx.from_pandas_edgelist(df, source=f"{src_tar_substr}1", target=f"{src_tar_substr}2", create_using=nx.Graph())
    except Exception as e:
        print(f'Error: An issue occurred attempting to read "{file_path}": {e}.')
        return None

    # Perform early return if data is not compatible
    if nx.is_directed(G):
        print("Error: The provided .csv graph must be undirected for this simulation.")
        return None

    if G.number_of_nodes() == 0:
        print("Error: The graph is empty.")
        return None

    if not G:
        print("Error: An issue arose attempting to read the provided .csv file. It may be incomplete.")
        return None

    return G


def print_comparison(stats_no_block: dict, stats_blocked: dict, total_nodes: int):
    """Print side-by-side comparison of cascade results."""
    print(f"\n{'='*60}")
    print("INTERVENTION COMPARISON")
    print(f"{'='*60}")
    print(f"Total nodes in graph:      {total_nodes}")

    dynamically_blocked = stats_blocked.get('dynamically_blocked', 0)
    print(f"Nodes blocked:             {dynamically_blocked} ({dynamically_blocked/total_nodes*100:.1f}%)")
    print(f"Strategy:                  DYNAMIC (degree-priority)")

    print(f"\n{'─'*60}")
    print("WITHOUT BLOCKING:")
    print(f"  Final adopters:          {stats_no_block['total_adopters']}")
    print(f"  Adoption rate:           {stats_no_block['adoption_rate']:.1%}")
    print(f"  Cascade rounds:          {stats_no_block['rounds']}")

    print(f"\n{'─'*60}")
    print("WITH DYNAMIC BLOCKING:")
    print(f"  Final adopters:          {stats_blocked['total_adopters']}")
    print(f"  Adoption rate:           {stats_blocked['adoption_rate']:.1%}")
    print(f"  Cascade rounds:          {stats_blocked['rounds']}")

    # Calculate reduction
    reduction = stats_no_block['total_adopters'] - stats_blocked['total_adopters']
    reduction_pct = (reduction / stats_no_block['total_adopters'] * 100) if stats_no_block['total_adopters'] > 0 else 0

    print(f"\n{'─'*60}")
    print("EFFECTIVENESS:")
    print(f"  Adopters prevented:      {reduction}")
    print(f"  Reduction:               {reduction_pct:.1f}%")
    print(f"{'='*60}")


def main():
    params = parse_args()

    # Extracts parsed arguments
    csv_file = params["csv_file_pathway"]
    color = params["color"]
    title = params["title"]
    sample_size = params["sample_size"]
    risk_assessment = params["risk_assessment"]
    bow_tie = params["bow_tie"]
    cascade = params["cascade"]
    block = params["block"]
    block_pct = params["block_pct"]
    block_per_round = params["block_per_round"]

    G = None

    # Check to ensure user-provided color is valid
    if color:
        color = validate_color(color)

    if csv_file:
        G = read_csv_data(csv_file)

        if not G:
            print("Error: Unable to read .csv file.")
            return

        # Remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))

        # Displays multiple communities with color-coding
        if "facebook_large" in csv_file:
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

        # Run Linear Threshold cascade from riskiest node
        if cascade:
            risk_scores = compute_misinformation_risk(H)
            riskiest_node = get_most_influential_node(H, risk_scores)

            if block:
                print(f"\n{'='*60}")
                print(f"DYNAMIC BLOCKING INTERVENTION")
                print(f"Budget: {block_pct*100:.1f}% of nodes")
                if block_per_round:
                    print(f"Per-round limit: {block_per_round} nodes")
                print(f"{'='*60}")

                # Run WITHOUT blocking first
                print("\n>>> Running cascade WITHOUT blocking...")
                stats_no_block = perform_cascade(H, threshold=0.1, initial_adopters=[riskiest_node], blocked_nodes=None)

                # Reset graph state for second run
                for node in H.nodes():
                    H.nodes[node]["adopter_round"] = float("inf")
                    H.nodes[node]["p"] = 0.0
                    H.nodes[node]["blocked"] = False

                # Run WITH dynamic blocking
                print("\n>>> Running cascade WITH dynamic blocking...")
                stats_blocked = perform_cascade(H, threshold=0.1, initial_adopters=[riskiest_node],
                                                blocked_nodes=set(), block_budget=block_pct,
                                                block_per_round=block_per_round)

                # Print comparison
                print_comparison(stats_no_block, stats_blocked, len(H.nodes()))

                # Animate the blocked version
                all_blocked = {n for n in H.nodes() if H.nodes[n].get("blocked", False)}
                print(f"\nSaving cascade animation to img/cascade_animation.gif...")
                animate_cascade(H, title="Cascade Spread (With Dynamic Blocking)", interval=600,
                               save_path="img/cascade_animation.gif", blocked_nodes=all_blocked)
            else:
                # Original behavior - no blocking
                perform_cascade(H, threshold=0.1, initial_adopters=[riskiest_node])
                print("Saving cascade animation to img/nonblock_cascade_animation.gif...")
                animate_cascade(H, title="Cascade Spread", interval=600, save_path="img/nonblock_cascade_animation.gif")
    else:
        print("Error: A .csv file was not provided.")
        return

    # Check to ensure .csv graph data is given
    if not G:
        print("Error: Unable to read .csv file.")
        return


if __name__ == "__main__":
    main()
