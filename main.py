import argparse
import sys
import networkx as nx
import pandas as pd
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
        "csv_file_pathway",
        type=str,
        help="The path to the inputted '.csv' graph file."
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


def verify_compatible_file(input_file: str, file_extension: str) -> bool:
    """
    Checks to see if the file inputted is compatible with the program. If not, forces early return.

    Args:
        input_file: The name of the '.csv' file containing graph data.
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
        file_path: The name of the '.csv' file containing graph data.
    """
    # Perform early return if file is not compatible
    if not verify_compatible_file(file_path, ".csv"):
        return None
    
    # If program reaches this point, the '.csv' file provided must be valid
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
        print("Error: The provided '.csv' graph must be undirected for this simulation.")
        return None
    
    if G.number_of_nodes() == 0:
        print("Error: The graph is empty.")
        return None
    
    if not G:
        print("Error: An issue arose attempting to read the provided '.csv' file. It may be incomplete.")
        return None
    
    return G


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
    sirs = params["sirs"]

    G = None

    # Check to ensure user-provided color is valid
    if color:
        color = validate_color(color)

    if csv_file:
        G = read_csv_data(csv_file)

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
                max_steps=50,
                seed=123,
                record_node_states=True   # REQUIRED for animation
            )

            animate_sirs(H, history["states"], save_path="img/sirs_simulation.gif")
    else:
        print("Error: A '.csv' file was not provided.")
        return

    
    # Check to ensure .csv graph data is given
    if not G:
        print("Error: Unable to read '.csv' file.")
        return


if __name__ == "__main__":
    main()
