# ----- Setup ----- #
import pandas as pd
import networkx as nx
import numpy as np
import os
import random as rd
import logging


# Initialise logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():  # Create handlers
    c_handler = logging.StreamHandler()  # Handler for writing to console
    f_handler = logging.FileHandler("file.log")  # Handler for writing to log file
    c_handler.setLevel(logging.WARNING)  # c_handler logs warning and higher
    f_handler.setLevel(logging.DEBUG)  # f_handler logs debug and higher

    # Create formatters and add them to handlers
    c_format = logging.Formatter("Module: %(name)s, function: %(funcName)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(levelname)s - Module: %(name)s, function: %(funcName)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

def change_logger(log_file_path: str = "file.log", 
                 console_level: int = logging.WARNING,
                 console_format: str = "Module: %(name)s, function: %(funcName)s - %(message)s",
                 file_level: int = logging.DEBUG,
                 file_format: str = "%(asctime)s - %(levelname)s - Module: %(name)s, function: %(funcName)s - %(message)s"
                ) -> None:
    """
    Change the logger with the given file path, console level, file level, and formats.

    ### Args:
        - log_file_path (str): Path for the log file. Default is current working directory.
        - console_level (int): Logging level for the console handler. Default is logging.WARNING.
        - console_format (str): Log format for the console handler. Default is "Module: %(name)s, function: %(funcName)s - %(message)s".
        - file_level (int): Logging level for the file handler. Default is logging.DEBUG.
        - file_format (str): Log format for the file handler. Default is "%(asctime)s - %(levelname)s - Module: %(name)s, function: %(funcName)s - %(message)s"

    ### Return:
        - None
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # Create handlers
    c_handler = logging.StreamHandler()  # Handler for writing to console
    f_handler = logging.FileHandler(log_file_path)  # Handler for writing to log file
    c_handler.setLevel(console_level)  # Set console handler log level
    f_handler.setLevel(file_level)  # Set file handler log level

    # Create formatters and add them to handlers
    c_format = logging.Formatter(console_format)
    f_format = logging.Formatter(file_format)
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)


standard_path = os.path.join(os.path.expanduser("~"), "Downloads") # Set standard path to save output files if no path is given

def set_standard_path(new_path: str) -> None:
    '''
    Set a new standard path for saving files globally.

    ### Args:
        - new_path (str): The new path to set as the standard path.

    ### Return:
        - None
    '''
    global standard_path
    standard_path = new_path
    logger.info(f"Standard path set to: {standard_path}")



# ----- General Calculations ----- #
def create_scalefree(max_nodes: int, step_edges: int, seed: int | list[int] | np.ndarray | None = None, average: float = 0.5) -> nx.Graph:
    '''
    Generate a scale-free network using the Barabási-Albert model and add random weights to the edges.

    ### Args:
        - max_nodes (int): The maximum number of nodes in the graph.
        - step_edges (int): The number of edges each new node creates when added to the graph.
        - seed (int | list[int] | np.ndarray | None, optional): A seed for the random number generator. Default is None.
        - average (float, optional): The mean of the normal distribution from which the edge weights are drawn. Default is 0.5.

    ### Return:
        - nx.Graph: A scale-free graph with weighted edges.
    '''
    # Set the seed for Numpy to allow deterministic outcomes
    if seed is not None:
        np.random.seed(seed)

    graph = nx.barabasi_albert_graph(max_nodes, step_edges, seed) # Creates the graph
    num_edges = graph.number_of_edges()

    logger.info(f"Created a Barabasi-Albert graph with {max_nodes} nodes and {num_edges} edges. No edge weights yet.")

    # Change the type of the node names to a string
    node_mapping = {i: str(i) for i in range(max_nodes)}
    graph = nx.relabel_nodes(graph, node_mapping)

    # Generate the weights
    weights = np.random.normal(loc=average, scale=0.2, size=num_edges)
    weights = np.clip(weights, 0.001, 0.999)  # Make sure the weights are in an interval of (0, 1)

    # Add weights to the edges
    for (u, v), weight in zip(graph.edges(), weights):
        graph[u][v]["weight"] = weight

    logger.info(f"Added the edge weights to the graph and returns it.")
    return graph


def input_file(path: str) -> nx.Graph:
    '''
    Read a CSV file, parse its content into a graph, and handle duplicate edges by keeping the edge with the higher weight.

    ### Args:
        - path (str): The path to the CSV file. The file should have columns 'source', 'target', and 'weight', where values are separated by ';' and decimals are indicated with '.'.

    Return:
        - nx.Graph: A graph constructed from the CSV file with weighted edges.
    '''
    filename = os.path.basename(path) # Extract the filename from the path
    df = pd.read_csv(path, sep=";", decimal=".", on_bad_lines="warn", dtype={ # Create DataFrame for easier handling of data
        "source": "string",
        "target": "string",
        "weight": "float64"
    })
    
    G = nx.Graph() # New empty graph
    count_dupes = 0 # Count the duplicate edges
    
    for row in df.itertuples(index=False): # Extract the information from each row, index made by nx is ignored
        source = row.source
        target = row.target
        weight = row.weight
    
        if G.has_edge(source, target): # Check if edge exists
            count_dupes += 1
            existing_weight = G[source][target]["weight"]
            if weight > existing_weight:
                G[source][target]["weight"] = weight # Update weight if bigger
        else:
            G.add_edge(source, target, weight=weight) # Else just add the edge to the graph
    logger.info(f"There were {count_dupes} duplicate edges in the file {filename}.")
    return G


def sloops(graph: nx.Graph) -> int:
    '''
    Calculate the number of self-loops in the given graph.

    ### Args:
        - graph (nx.Graph): The graph in which to count self-loops.

    ### Return:
        - int: The number of self-loops in the graph.
    '''
    num_sloops = nx.number_of_selfloops(graph)
    logger.info(f"There are {num_sloops} self-loops in the network.")
    return num_sloops


def remove_sloops(graph: nx.Graph) -> nx.Graph:
    '''
    Remove all self-loops from the given graph.

    ### Args:
        - graph (nx.Graph): The graph from which to remove self-loops.

    ### Return:
        - nx.Graph: The graph with all self-loops removed.
    '''
    sloops = list(nx.selfloop_edges(graph))
    graph.remove_edges_from(sloops)
    logger.info(f"Removed all self-loops from the network.")
    return graph


def general_info(graph: nx.Graph) -> tuple[int, int]:
    '''
    Get the number of nodes and edges in the given graph.

    ### Args:
        - graph (nx.Graph): The graph for which to retrieve the number of nodes and edges.

    ### Return:
        - (int, int): A tuple containing the number of nodes and the number of edges in the graph.
    '''
    num_n = graph.number_of_nodes()
    num_e = graph.number_of_edges()
    logger.info(f"The given graph has {num_n} nodes and {num_e} edges.")
    return num_n, num_e


def components(graph: nx.Graph) -> list[set[str]]:
    '''
    Create a list containing each connected component of the graph as a set of nodes.

    ### Args:
        - graph (nx.Graph): The graph for which to find connected components.

    ### Return:
        - list[set[str]]: A list of sets, each containing the nodes of a connected component of the graph.
    '''
    connected_comps = sorted(list(nx.connected_components(graph)), key=len, reverse=True)
    logger.info(f"Retrieved the list of components. Found {len(connected_comps)} components.")
    return connected_comps


def comp_len(graph: nx.Graph) -> list[int]:
    '''
    Calculate the length of each connected component in the graph.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the length of each connected component.

    ### Return:
        - list[int]: A list of lengths of each connected component in the graph.
    '''
    len_comps = [len(c) for c in components(graph)] # Calculate the length of each of the components
    logger.info(f"The components have lengths {len_comps}.")
    return len_comps


def subgraphs(graph: nx.Graph) -> list[nx.Graph]:
    '''
    Create a list containing each connected component of the graph as a subgraph.

    ### Args:
        - graph (nx.Graph): The graph for which to find and create subgraphs.

    ### Return:
        - list[nx.Graph]: A list of subgraphs, each corresponding to a connected component of the original graph.
    '''
    subgraphs = []
    connected_comps = components(graph)
    for comp in connected_comps:
        subgraph = graph.subgraph(comp)
        subgraphs.append(subgraph)
    logger.info(f"Created a list containing the subgraphs of the given graph. Created {len(subgraphs)} subgraphs.")
    return subgraphs


def all_shortest_paths(graph: nx.Graph, save_path: str | None = None) -> dict[str, dict[str, dict[float, list[str]]]]:
    '''
    Calculate all shortest paths between two distinct nodes and optionally save them in a CSV file.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the shortest paths.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - dict[str, dict[str, dict[float, list[str]]]]: A nested dictionary containing the lengths and paths of the shortest paths for each pair of nodes.
        First string is node 1, second string is node 2, float is the distance, list[str] is the path
    '''
    shortest_paths = dict(nx.shortest_path(graph, weight="weight")) # Calculate all shortest paths
    df_data = []
    paths = {}
    
    for source, targets in shortest_paths.items(): # Iterate through all edges of the graph
        paths[source] = {}
        for target, path in targets.items():
            # Calculate the length of the path based on the weights of the edges
            length = sum(graph[path[i]][path[i+1]]["weight"] for i in range(len(path) - 1))
            # Append the data to the list for CSV export
            df_data.append({"source": source, "target": target, "length": length, "path": path})
            # Store the path length and path in the nested dictionary
            paths[source][target] = {"length": length, "path": path}
    logger.info(f"Iterated over all edges and added shortest paths to the dictionary.")
    if save_path is not None:
        filename = "shortest_paths.csv"
        full_path = os.path.join(save_path, filename)
        df = pd.DataFrame(df_data)
        logger.info(f"Created DataFrame, trying to save to CSV.")
        df.to_csv(full_path, index=False)
        logger.info(f"Saved the file here: {full_path}")
        
    return paths


def calc_cc(graph: nx.Graph, path: str | None = None) -> dict[str, float]:
    '''
    Calculate the clustering coefficients for all nodes in the graph and optionally save them to a CSV file.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the clustering coefficients.
        - path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - dict[str, float]: A dictionary where the keys are node labels and the values are the clustering coefficients.
    '''
    # Calculate clustering coefficients for all nodes in the graph
    clustering_coeffs = nx.clustering(graph, weight="weight")

    # If a path is provided, save the clustering coefficients to a CSV file
    if path is not None:
        filename = "clustering_coefficients.csv"
        full_path = os.path.join(path, filename)
        cc = pd.DataFrame(clustering_coeffs.items(), columns=["Node", "Clustering Coefficient"]) # Convert clustering coefficients to DataFrame
        logger.info(f"Created DataFrame, trying to save to CSV.")
        cc.to_csv(full_path, index=False, sep=";") # Save the DataFrame to a CSV file
        logger.info(f"Saved the file here: {full_path}")
        
    return clustering_coeffs


def sw_coeff(graph: nx.Graph, func: str) -> tuple[list[float], float]:
    '''
    Calculate the small-world coefficient sigma or omega for each component and the weighted average. 

    The weighted average is calculated by multiplying the calculated coefficient for each subgraph by the size of the subgraph for a weighted total and then dividing by the number of subgraphs.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the small-world coefficients.
        - func (str): The type of small-world coefficient to calculate, either 'sigma' or 'omega'.

    ### Return:
        - tuple[list[float], float]: A tuple containing a list of small-world coefficients for each component and the weighted average of these coefficients.
    '''
    total_size = len(graph)  # Total number of nodes in the graph
    weighted_total = 0.0  # To accumulate the weighted sum of coefficients
    sw_coeffs = []  # List to store the small-world coefficients for each subgraph

    # Validate the input function type
    if func not in {"sigma", "omega"}:
        logger.error(f"Please choose either 'sigma' or 'omega'.")
        raise TypeError(f"Invalid function type provided: {func}")
    
    # Iterate over each subgraph (connected component)
    for subgraph in subgraphs(graph):
        sg_size = len(subgraph)  # Size of the current subgraph
        # Calculate the small-world coefficient based on the specified function
        if func == "sigma":
            sg_coeff = nx.sigma(subgraph)
        elif func == "omega":
            sg_coeff = nx.omega(subgraph)
        
        sw_coeffs.append(sg_coeff)  # Append the coefficient to the list
        weighted_total += sg_size * sg_coeff  # Accumulate the weighted total
    
    # Calculate the weighted average of the small-world coefficients
    weighted_average = weighted_total / total_size
    logger.info(f"Calculated a weighted total of {weighted_total} and a total size of {total_size}. The weighted average therefore is {weighted_average}.")
    
    return sw_coeffs, weighted_average


def modularity(graph: nx.Graph) -> float:
    '''
    Return the modularity of a list of communities using the Clauset-Newman-Moore algorithm.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the modularity.

    ### Return:
        - float: The modularity of the graph based on the identified communities. 0 if graph has no nodes.
    '''
    # Check if the graph is empty
    if len(graph.nodes) == 0:
        logger.info(f"The graph has no nodes, returned a modularity of 0.")
        return 0
    
    # Identify communities using the Clauset-Newman-Moore greedy modularity maximization algorithm
    communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
    logger.info(f"Identified {len(communities)} communities. Calculating modularity based on found community structure.")
    
    # Calculate the modularity of the identified communities
    modularity = nx.community.modularity(graph, communities)
    logger.info(f"Calculated a modularity of {modularity}.")
    
    return modularity


def communities(graph: nx.Graph) -> list[list[str]]:
    '''
    Return a list of communities using the Clauset-Newman-Moore algorithm.

    ### Args:
        - graph (nx.Graph): The graph for which to identify the communities.

    ### Return:
        - list[list[str]]: A list of communities, where each community is represented as a list of node labels.
    '''
    # Check if the graph is empty
    if len(graph.nodes) == 0:
        logger.info(f"The graph has no nodes, returned an empty list.")
        return []
    
    # Identify communities using the Clauset-Newman-Moore greedy modularity maximization algorithm
    communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
    logger.info(f"Identified {len(communities)} communities and returned them as a list.")
    return communities


def select_random_nodes(graph: nx.Graph, percent: int, seed: int | float | str | bytes | bytearray | None = None) -> list[str]:
    '''
    Select random nodes from the graph based on a given percentage of the total nodes.

    ### Args:
        - graph (nx.Graph): The graph from which to select nodes.
        - percent (int): The percentage of nodes to select.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - list[str]: A list of randomly selected node labels.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility

    # Convert the nodes of the graph to a list
    node_list = list(graph.nodes())
    
    # Determine the number of nodes to select
    num_nodes = int(len(node_list) * (percent / 100))
    
    # Select random nodes from the list
    random_nodes = rd.sample(node_list, num_nodes)
    logger.info(f"Chose {num_nodes} many random nodes and returned them as a list")
    
    return random_nodes



# ----- Averages ----- #
def avg_shortest_path_len(graph):
    total_length = 0
    total_nodes = 0
    
    for subgraph in subgraphs(graph):
        num_nodes = subgraph.number_of_nodes()
        
        
        if subgraph.number_of_edges() > 0:
            avg_length = nx.average_shortest_path_length(subgraph, weight="weight")
            total_length += avg_length * num_nodes
            total_nodes += num_nodes
    
    if total_nodes == 0:
        return 0 
    
    return total_length / total_nodes


def avg_shortest_path_len(graph: nx.Graph) -> float:
    '''
    Calculate the average shortest path length for the entire graph, taking into account each connected component.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the average shortest path length.

    ### Return:
        - float: The average shortest path length of the graph.
    '''
    total_length = 0  # To accumulate the total length of all shortest paths
    total_nodes = 0   # To accumulate the total number of nodes
    
    # Iterate over each subgraph
    for subgraph in subgraphs(graph):
        num_nodes = subgraph.number_of_nodes()
        
        # Check if the subgraph has any edges
        if subgraph.number_of_edges() > 0:
            avg_length = nx.average_shortest_path_length(subgraph, weight="weight")
            total_length += avg_length * num_nodes # Weight the average length by the number of nodes
            total_nodes += num_nodes
    
    # If there are no nodes with edges, return 0
    if total_nodes == 0:
        logger.info(f"The graph has no nodes, returned 0 as average.")
        return 0 
    
    # Calculate the weighted average shortest path length
    average_length = total_length / total_nodes
    logger.info(f"Iterated over all subgraphs. Calculated a weighted total length of {total_length} and a total of {total_nodes} nodes and therefore an average of {average_length}.")
    return average_length


def density(graph: nx.Graph) -> float:
    '''
    Return the density of the graph.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the density.

    ### Return:
        - float: The density of the graph.
    '''
    density = nx.density(graph)
    logger.info(f"Calculated a density of {density}.")
    return density


def average_degree(graph: nx.Graph) -> float:
    '''
    Calculate the average degree of the nodes in the network.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the average degree.

    ### Return:
        - float: The average degree of the nodes in the graph.
    '''
    num_n = graph.number_of_nodes()  # Total number of nodes in the graph
    num_e = graph.number_of_edges()  # Total number of edges in the graph
    logger.info(f"The graph has {num_n} nodes and {num_e} edges. Calculating average degree.")
    
    # Calculate the average degree using the formula (2 * number of edges) / number of nodes
    avg_degree = (2 * num_e) / num_n
    logger.info(f"The graph has an average degree of {avg_degree}.")
    return avg_degree


def avg_weight(graph: nx.Graph | pd.DataFrame) -> float:
    '''
    Calculate the average weight of edges in the graph or DataFrame.

    ### Args:
        - graph (nx.Graph | pd.DataFrame): The graph or DataFrame for which to calculate the average weight. If a DataFrame is provided, it should have a "weight" column.

    ### Return:
        - float: The average weight of the edges.
    '''
    # Calculate the total weight and number of edges based on the input type
    if isinstance(graph, nx.Graph):
        total_weight = graph.size(weight="weight")  # Total weight of all edges in the graph
        num_edges = nx.number_of_edges(graph)  # Total number of edges in the graph
        logger.info(f"Input was nx.Graph. Calculated a total weight of {total_weight} and {num_edges} edges.")
    elif isinstance(graph, pd.DataFrame):
        logger.info(f"Input was pd.DataFrame. Calculated a total weight of {total_weight} and {num_edges} edges.")
        total_weight = sum(graph["weight"])  # Sum of weights in the DataFrame
        num_edges = len(graph)  # Number of rows in the DataFrame (each representing an edge)
    
    # Calculate the average weight
    average_weight = total_weight / num_edges
    logger.info(f"Calculated an average weight of {average_weight}.")
    
    return average_weight


def average_cc(data: (dict | str)) -> float:
    '''
    Calculate the average of the local clustering coefficients from a dictionary or a CSV file.

    ### Args:
        - data (dict | str): The input data. It can be a dictionary of clustering coefficients or a file path to a CSV file.

    ### Return:
        - float: The average clustering coefficient.
    '''
    # Check if the input is a dictionary of clustering coefficients
    if isinstance(data, dict):
        logger.info(f"Input was a dictionary. Calculating average clustering coefficient.")
        avg_cc = sum(data.values()) / len(data)  # Calculate average clustering coefficient from dictionary
    # Check if the input is a file path to a CSV file
    elif isinstance(data, str):
        logger.info(f"Input was a path. Calculating average clustering coefficient.")
        cc_df = pd.read_csv(data, sep=";")  # Read the CSV file
        avg_cc = cc_df["Clustering Coefficient"].mean()  # Calculate average clustering coefficient from DataFrame
    else:
        logger.error(f"Could not calculate the average clustering coefficient because of an invalid input.")
        raise TypeError("Input should be either a dictionary of clustering coefficients or a file path to a CSV file.")
    
    logger.info(f"Calculated an average clustering coefficient of {avg_cc}.")
    return avg_cc



# ----- Centralities + Centers ----- #
def save_to_csv(dataframe: pd.DataFrame, filename: str, save_path: str = standard_path) -> None:
    '''
    Save a DataFrame to a CSV file.

    ### Args:
        - dataframe (pd.DataFrame): The DataFrame to save.
        - filename (str): The name of the file to save the DataFrame to.
        - save_path (str, optional): The directory path where the CSV file will be saved. Default is standard_path.

    ### Return:
        - None
    '''
    # Construct the full file path
    full_path = os.path.join(save_path, filename)
    
    # Save the DataFrame to a CSV file
    dataframe.to_csv(full_path, index=False)
    logger.info(f"Saved the file {filename} as {full_path}.")


def degree_ct(graph: nx.Graph, rnd: int = 10, path: str | None = None) -> dict[str, float]:
    '''
    Calculate the degree centrality of the graph, optionally save the results to a CSV file, and return the centrality values.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate degree centrality.
        - rnd (int, optional): The number of decimal places to round the degree centrality values to when saving as CSV. Default is 10.
        - path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - dict[str, float]: A dictionary of degree centrality values for each node.
    '''
    # Calculate degree centrality for each node
    dg_ct = nx.degree_centrality(graph)
    logger.info(f"Calculated the degree centralities.")

    # Save the DataFrame to a CSV file if a valid path is provided
    if path is not None:
        df = pd.DataFrame(dg_ct.items(), columns=["Node", "Degree Centrality"]) # Create a DataFrame from the degree centrality dictionary
        df["Degree Centrality"] = df["Degree Centrality"].round(rnd) # Round the degree centrality values to the specified number of decimal places
        filename = "degree_centrality.csv"
        save_to_csv(dataframe=df, filename=filename, save_path=path)

    logger.info(f"Returning the degree centralities as a dict.")
    return dg_ct


def eigenvector_ct(graph: nx.Graph, rnd: int = 10, path: str | None = None) -> dict[str, float]:
    '''
    Calculate the eigenvector centrality of the graph, optionally save the results to a CSV file, and return the centrality values.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate eigenvector centrality.
        - rnd (int, optional): The number of decimal places to round the eigenvector centrality values to when saving as a CSV. Default is 10.
        - path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - dict[str, float]: A dictionary of eigenvector centrality values for each node.
    '''
    # Calculate eigenvector centrality for each node
    ev_ct = nx.eigenvector_centrality_numpy(graph, weight="weight")
    logger.info(f"Calculated the eigenvector centralities.")
    
    # Save the DataFrame to a CSV file if a valid path is provided
    if path is not None:
        df = pd.DataFrame(ev_ct.items(), columns=["Node", "Eigenvector Centrality"]) # Create a DataFrame from the eigenvector centrality dictionary
        df["Eigenvector Centrality"] = df["Eigenvector Centrality"].round(rnd) # Round the eigenvector centrality values to the specified number of decimal places
        filename = "eigenvector_centrality.csv"
        save_to_csv(dataframe=df, filename=filename, save_path=path)

    logger.info(f"Returning the eigenvector centralities as a dict.")
    return ev_ct


def closeness_ct(graph: nx.Graph, rnd: int = 10, path: str | None = None) -> dict[str, float]:
    '''
    Calculate the closeness centrality of the graph, optionally save the results to a CSV file, and return the centrality values.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate closeness centrality.
        - rnd (int, optional): The number of decimal places to round the closeness centrality values to when saving as a CSV. Default is 10.
        - path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - dict[str, float]: A dictionary of closeness centrality values for each node.
    '''
    # Calculate closeness centrality for each node
    cn_ct = nx.closeness_centrality(graph, distance="weight")
    logger.info(f"Calculated the closeness centralities.")
    
    # Save the DataFrame to a CSV file if a valid path is provided
    if path is not None:
        df = pd.DataFrame(cn_ct.items(), columns=["Node", "Closeness Centrality"])  # Create a DataFrame from the closeness centrality dictionary
        df["Closeness Centrality"] = df["Closeness Centrality"].round(rnd)  # Round the closeness centrality values to the specified number of decimal places
        filename = "closeness_centrality.csv"
        save_to_csv(dataframe=df, filename=filename, save_path=path)

    logger.info(f"Returning the closeness centralities as a dict.")
    return cn_ct


def pgrank(graph: nx.Graph, rnd: int = 10, path: str | None = None) -> dict[str, float]:
    '''
    Calculate the PageRank of the graph, optionally save the results to a CSV file, and return the PageRank values.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate PageRank.
        - rnd (int, optional): The number of decimal places to round the PageRank values to when saving as a CSV. Default is 10.
        - path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - dict[str, float]: A dictionary of PageRank values for each node.
    '''
    # Calculate PageRank for each node
    pgrank = nx.pagerank(graph)
    logger.info(f"Calculated the PageRank values.")
    
    # Save the DataFrame to a CSV file if a valid path is provided
    if path is not None:
        df = pd.DataFrame(pgrank.items(), columns=["Node", "Pagerank"])  # Create a DataFrame from the PageRank dictionary
        df["Pagerank"] = df["Pagerank"].round(rnd)  # Round the PageRank values to the specified number of decimal places
        filename = "pagerank.csv"
        save_to_csv(dataframe=df, filename=filename, save_path=path)

    logger.info(f"Returning the PageRank values as a dict.")
    return pgrank


def node_betweenness_ct(graph: nx.Graph, rnd: int = 10, path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> dict[str, float]:
    '''
    Calculate the node betweenness centrality of the graph, optionally save the results to a CSV file, and return the centrality values.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate node betweenness centrality.
        - rnd (int, optional): The number of decimal places to round the node betweenness centrality values to when saving as a CSV. Default is 10.
        - path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): Seed for random number generator. Default is None.

    ### Return:
        - dict[str, float]: A dictionary of node betweenness centrality values for each node.
    '''
    nd_bt_ct = {}  # Initialize an empty dictionary to store node betweenness centrality values
    
    # Calculate betweenness centrality
    if nx.is_connected(graph):
        nd_bt_ct = nx.betweenness_centrality(graph, weight="weight", seed=seed)
    else:
        for subgraph in subgraphs(graph):
            nd_bt_subgraph = nx.betweenness_centrality(subgraph, weight="weight", seed=seed)
            for node, centrality in nd_bt_subgraph.items():
                nd_bt_ct[node] = centrality  # Add the centrality value to the dictionary

    logger.info("Calculated the node betweenness centralities.")

    # Save the DataFrame to a CSV file if a valid path is provided
    if path is not None:
        df = pd.DataFrame(nd_bt_ct.items(), columns=["Node", "Node Betweenness Centrality"])  # Create a DataFrame from the node betweenness centrality dictionary
        df["Node Betweenness Centrality"] = df["Node Betweenness Centrality"].round(rnd)  # Round the node betweenness centrality values to the specified number of decimal places
        filename = "node_betweenness.csv"
        save_to_csv(dataframe=df, filename=filename, save_path=path)

    logger.info(f"Returning the node betweenness values as a dict.")
    return nd_bt_ct


def edge_betweenness_ct(graph: nx.Graph, rnd: int = 10, path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> dict[tuple[str, str], float]:
    '''
    Calculate the edge betweenness centrality of the graph, optionally save the results to a CSV file, and return the centrality values.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate edge betweenness centrality.
        - rnd (int, optional): The number of decimal places to round the edge betweenness centrality values to when saving as a CSV. Default is 10.
        - path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): Seed for random number generator. Default is None.

    ### Return:
        - dict[tuple[str, str], float]: A dictionary of edge betweenness centrality values for each edge.
    '''
    # Calculate edge betweenness centrality for each edge
    ed_bt_ct = nx.edge_betweenness_centrality(graph, weight="weight", seed=seed)
    logger.info("Calculated the edge betweenness centralities.")
    
    # Save the DataFrame to a CSV file if a valid path is provided
    if path is not None:
        df = pd.DataFrame(ed_bt_ct.items(), columns=["Edge", "Edge Betweenness Centrality"])  # Create a DataFrame from the edge betweenness centrality dictionary
        df["Edge Betweenness Centrality"] = df["Edge Betweenness Centrality"].round(rnd)  # Round the edge betweenness centrality values to the specified number of decimal places
        filename = "edge_betweenness_centrality.csv"
        save_to_csv(dataframe=df, filename=filename, save_path=path)

    logger.info(f"Returning the edge betweenness values as a dict.")
    return ed_bt_ct


def find_eccentricity(graph: nx.Graph, shrtst_paths: dict[str, dict[str, float]] | None = None) -> list[dict[str, float]]:
    '''
    Calculate the eccentricity for each node in each subgraph of the graph.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the eccentricity.
        - shrtst_paths (dict[str, dict[str, float]] | None, optional): All pairs shortest path lengths. Default is None.

    ### Return:
        - list[dict[str, float]]: A list of dictionaries, containing the eccentricity values for the nodes in each subgraph.
    '''
    eccs = []  # Initialize a list to store eccentricity dictionaries for each subgraph
    
    # Iterate over each subgraph
    for subgraph in subgraphs(graph):
        # Calculate the eccentricity for each node in the subgraph
        ecc = nx.eccentricity(G=subgraph, weight="weight", sp=shrtst_paths)
        eccs.append(ecc)  # Append the eccentricity dictionary to the list

    logger.info(f"Found {len(eccs)} subgraphs and returned the eccentricity values as a list.")
    return eccs


def find_center(graph: nx.Graph, eccs: list[dict[str, float]] | None = None) -> list[list[str]]:
    '''
    Calculate the center for each subgraph of the graph.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the center.
        - eccs (list[dict[str, float]] | None, optional): Precomputed eccentricities for the subgraphs. Default is None.

    ### Return:
        - list[list[str]]: A list of lists, each containing the nodes in the center of a subgraph.
    '''
    centers = []  # Initialize a list to store the centers of each subgraph
    
    if eccs is None: # If no precomputed eccentricities are provided
        logger.info(f"No precomputed eccentricities were provided. Now calculating centers.")
        for subgraph in subgraphs(graph):
            center = nx.center(subgraph, weight="weight") # Calculate the center of the subgraph
            centers.append(center)
    else: # If precomputed eccentricities are provided
        logger.info(f"Precomputed eccentricities were provided. Now calculating centers using the provided eccentricities.")
        for subgraph, ecc in zip(subgraphs(graph), eccs):
            center = nx.center(G=subgraph, e=ecc, weight="weight") # Calculate the center of the subgraph using the provided eccentricities
            centers.append(center)

    logger.info(f"Calculated the center nodes for {len(centers)} subgraphs and returned them as a list.")
    return centers


def find_barycenter(graph: nx.Graph) -> list[list[str]]:
    '''
    Calculate the barycenter for each subgraph of the graph.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the barycenter.

    ### Return:
        - list[list[str]]: A list of lists, each containing the nodes in the barycenter of a subgraph.
    '''
    centers = []  # Initialize a list to store the barycenters of each subgraph
    
    for subgraph in subgraphs(graph):
        center = nx.barycenter(subgraph, weight="weight") # Calculate the barycenter of the subgraph
        centers.append(center)
    logger.info(f"Calculated the barycenter nodes for {len(centers)} subgraphs and returned them as a list.")
    return centers


def diam(graph: nx.Graph, eccs: list[dict[str, float]] | None = None) -> list[float]:
    '''
    Calculate the diameter for each subgraph of the graph.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the diameter.
        - eccs (list[dict[str, float]] | None, optional): Precomputed eccentricities for the subgraphs. Default is None.

    ### Return:
        - list[float]: A list of diameters for each subgraph.
    '''
    diams = []  # Initialize a list to store the diameters of each subgraph
    
    if eccs is None: # If no precomputed eccentricities are provided
        logger.info(f"No precomputed eccentricities were provided. Now calculating diameter.")
        for subgraph in subgraphs(graph):
            diam = nx.diameter(subgraph, weight="weight") # Calculate the diameter of the subgraph
            diams.append(diam)
    else: # If precomputed eccentricities are provided
        logger.info(f"Precomputed eccentricities were provided. Now calculating diameter using the provided eccentricities.")
        for subgraph, ecc in zip(subgraphs(graph), eccs):
            diam = max(ecc.values()) # Calculate the diameter of the subgraph using the provided eccentricities
            diams.append(diam)

    logger.info(f"Calculated the diameter for {len(diams)} subgraphs and returned them as a list.")
    return diams


def kem_c(graph: nx.Graph) -> list[float]:
    '''
    Calculate the Kemeny constant for each subgraph of the graph.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the Kemeny constant.

    ### Return:
        - list[float]: A list of Kemeny constants for each subgraph.
    '''
    kem_cs = []  # Initialize a list to store the Kemeny constants of each subgraph
    
    for subgraph in subgraphs(graph):
        kem_c = nx.kemeny_constant(subgraph, weight="weight") # Calculate the Kemeny constant of the subgraph
        kem_cs.append(kem_c)
    
    logger.info(f"Calculated the Kemeny constant for {len(kem_cs)} subgraphs and returned them as a list.")
    return kem_cs


def periphery(graph: nx.Graph, eccs: list[dict[str, float]] | None = None) -> list[list[str]]:
    '''
    Calculate the periphery for each subgraph of the graph.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the periphery.
        - eccs (list[dict[str, float]] | None, optional): Precomputed eccentricities for the subgraphs. Default is None.

    ### Return:
        - list[list[str]]: A list of lists, each containing the nodes in the periphery of a subgraph.
    '''
    periphs = []  # Initialize a list to store the peripheries of each subgraph
    
    if eccs is None: # If no precomputed eccentricities are provided
        logger.info(f"No precomputed eccentricities were provided. Now calculating peripheries.")
        for subgraph in subgraphs(graph):
            periph = nx.periphery(subgraph, weight="weight") # Calculate the periphery of the subgraph
            periphs.append(periph)
    else: # If precomputed eccentricities are provided
        logger.info(f"Precomputed eccentricities were provided. Now calculating peripheries using the provided eccentricities.")
        for subgraph, ecc in zip(subgraphs(graph), eccs):
            periph = nx.periphery(subgraph, e=ecc, weight="weight") # Calculate the periphery of the subgraph using the provided eccentricities
            periphs.append(periph)
    
    logger.info(f"Calculated the periphery for {len(periphs)} subgraphs and returned them as a list.")
    return periphs


def radius(graph: nx.Graph, eccs: list[dict[str, float]] | None = None) -> list[float]:
    '''
    Calculate the radius for each subgraph of the graph.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the radius.
        - eccs (list[dict[str, float]] | None, optional): Precomputed eccentricities for the subgraphs. Default is None.

    ### Return:
        - list[float]: A list of radii for each subgraph.
    '''
    radii = []  # Initialize a list to store the radii of each subgraph
    
    if eccs is None: # If no precomputed eccentricities are provided
        logger.info(f"No precomputed eccentricities were provided. Now calculating radii.")
        for subgraph in subgraphs(graph):
            radius = nx.radius(subgraph, weight="weight") # Calculate the radius of the subgraph
            radii.append(radius)
    else: # If precomputed eccentricities are provided
        logger.info(f"Precomputed eccentricities were provided. Now calculating radii using the provided eccentricities.")
        for subgraph, ecc in zip(subgraphs(graph), eccs):
            radius = min(ecc.values()) # Calculate the radius of the subgraph using the provided eccentricities
            radii.append(radius)
    
    logger.info(f"Calculated the radius for {len(radii)} subgraphs and returned them as a list.")
    return radii


def resistance_distance(graph: nx.Graph) -> list[dict[tuple[str, str], float]]:
    '''
    Calculate the resistance distance for each subgraph of the graph.

    ### Args:
        - graph (nx.Graph): The graph for which to calculate the resistance distance.

    ### Return:
        - list[dict[tuple[str, str], float]]: A list of dictionaries, each one containing the resistance distance for the edges in the subgraph.
    '''
    all_dis = []  # Initialize a list to store the resistance distances of each subgraph
    
    for subgraph in subgraphs(graph):
        one_dis = nx.resistance_distance(subgraph, weight="weight") # Calculate the resistance distance for the subgraph
        all_dis.append(one_dis)

    logger.info(f"Calculated the radius for {len(all_dis)} subgraphs and returned them as a list.")
    return all_dis



# ----- Remove nodes and edges ----- #
def remove_nodes(graph: nx.Graph, nodes: list[str], filename: str = "modified_graph.csv", save_path: str | None = None) -> nx.Graph:
    '''
    Remove specified nodes from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which to remove nodes.
        - nodes (list[str]): A list of nodes to be removed.
        - filename (str, optional): The name of the file to save the modified graph. Default is "modified_graph.csv".
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - nx.Graph: The modified graph with specified nodes removed.
    '''
    modified_graph = graph.copy()  # Make a copy of the graph

    total_nodes = len(nodes)
    logger.info(f"Made a copy of the given graph. Will remove a total of {total_nodes} nodes.")

    # Remove the specified nodes
    modified_graph.remove_nodes_from(nodes)
    
    # Save modified graph to a CSV file if a valid path is provided
    if save_path is not None:
        logger.info(f"A path was given, trying to save the graph to a CSV file.")
        edge_list = []
        for u, v, d in modified_graph.edges(data=True):
            edge_list.append((u, v, d["weight"]))
        df = pd.DataFrame(edge_list, columns=["source", "target", "weight"])
        save_to_csv(dataframe=df, filename=filename, save_path=save_path)

    logger.info(f"Returned the modified graph.")
    return modified_graph


def remove_edges(graph: nx.Graph, edges: list[tuple[str, str]], filename: str = "modified_graph.csv", save_path: str | None = None) -> nx.Graph:
    '''
    Remove specified edges from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which to remove edges.
        - edges (list[tuple[str, str]]): A list of edges to be removed.
        - filename (str, optional): The name of the file to save the modified graph. Default is "modified_graph.csv".
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - nx.Graph: The modified graph with specified edges removed.
    '''
    modified_graph = graph.copy()  # Make a copy of the graph

    total_edges = len(edges)
    logger.info(f"Made a copy of the given graph. Will remove a total of {total_edges} edges.")
    
    # Remove the specified edges
    modified_graph.remove_edges_from(edges)
    
    # Save modified graph to a CSV file if a valid path is provided
    if save_path is not None:
        logger.info(f"A path was given, trying to save the graph to a CSV file.")
        edge_list = []
        for u, v, d in modified_graph.edges(data=True):
            edge_list.append((u, v, d["weight"]))
        df = pd.DataFrame(edge_list, columns=["source", "target", "weight"])
        save_to_csv(dataframe=df, filename=filename, save_path=save_path)
    
    logger.info(f"Returned the modified graph.")
    return modified_graph


def remove_center(graph: nx.Graph, save_path: str | None = None) -> nx.Graph:
    '''
    Remove the center nodes from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which to remove center nodes.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - nx.Graph: The modified graph with center nodes removed.
    '''
    centers = find_center(graph)
    centers = [node for sublist in centers for node in sublist] # Flatten the list of lists to a single list of nodes
    
    logger.info(f"Removing {len(centers)} center nodes.")
    removed_center = remove_nodes(graph=graph, nodes=centers, filename="without_center.csv", save_path=save_path)
    
    logger.info(f"Removed the center nodes and returned the modified graph.")
    return removed_center


def remove_barycenter(graph: nx.Graph, save_path: str | None = None) -> nx.Graph:
    '''
    Remove the barycenter nodes from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which to remove barycenter nodes.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - nx.Graph: The modified graph with barycenter nodes removed.
    '''
    barycenters = find_barycenter(graph)
    barycenters = [node for sublist in barycenters for node in sublist]  # Flatten the list of lists to a single list of nodes
    
    logger.info(f"Removing {len(barycenters)} barycenter nodes.")
    removed_barycenter = remove_nodes(graph=graph, nodes=barycenters, filename="without_barycenter.csv", save_path=save_path)
    
    logger.info(f"Removed the barycenter nodes and returned the modified graph.")
    return removed_barycenter


def remove_periphery(graph: nx.Graph, save_path: str | None = None) -> nx.Graph:
    '''
    Remove the periphery nodes from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which to remove periphery nodes.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - nx.Graph: The modified graph with periphery nodes removed.
    '''
    peripheries = periphery(graph)
    peripheries = [node for sublist in peripheries for node in sublist]  # Flatten the list of lists to a single list of nodes
    
    logger.info(f"Removing {len(peripheries)} periphery nodes.")
    removed_periphery = remove_nodes(graph=graph, nodes=peripheries, filename="without_periphery.csv", save_path=save_path)
    
    logger.info(f"Removed the periphery nodes and returned the modified graph.")
    return removed_periphery


def remove_random_nodes(graph: nx.Graph, number: int, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Remove a specified number of random nodes from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which to remove random nodes.
        - number (int): The number of random nodes to remove.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with random nodes removed.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility

    random_nodes = rd.sample(list(graph.nodes()), number)
    
    logger.info(f"Removing {number} random nodes: {random_nodes}.")
    removed_random_nodes = remove_nodes(graph=graph, nodes=random_nodes, filename="without_rd_nodes.csv", save_path=save_path)
    
    logger.info(f"Removed the random nodes and returned the modified graph.")
    return removed_random_nodes


def remove_top_degree_ct_nodes(graph: nx.Graph, number: int, save_path: str | None = None, dg_ct: dict[str, float] | None = None) -> nx.Graph:
    '''
    Remove the top nodes by degree centrality from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which to remove top degree centrality nodes.
        - number (int): The number of top nodes by degree centrality to remove.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - dg_ct (dict[str, float] | None, optional): Precomputed degree centrality values. Default is None.

    ### Return:
        - nx.Graph: The modified graph with top degree centrality nodes removed.
    '''
    if dg_ct is None:
        dg_ct = degree_ct(graph)
    
    # Sort the degree centrality dictionary by centrality values in descending order
    sorted_dg_ct = {node: centrality for node, centrality in sorted(dg_ct.items(), key=lambda item: item[1], reverse=True)}
    top_nodes = list(sorted_dg_ct.keys())[:number]  # Get the top 'number' nodes
    
    logger.info(f"Removing {number} top degree centrality nodes: {top_nodes}.")
    removed_top_nodes = remove_nodes(graph=graph, nodes=top_nodes, filename="without_degree_ct.csv", save_path=save_path)
    
    logger.info(f"Removed the top degree centrality nodes and returned the modified graph.")
    return removed_top_nodes


def remove_top_pgrank_nodes(graph: nx.Graph, number: int, save_path: str | None = None, pagerank: dict[str, float] | None = None) -> nx.Graph:
    '''
    Remove the top nodes by PageRank from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which to remove top PageRank nodes.
        - number (int): The number of top nodes by PageRank to remove.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - pagerank (dict[str, float] | None, optional): Precomputed PageRank values. Default is None.

    ### Return:
        - nx.Graph: The modified graph with top PageRank nodes removed.
    '''
    if pagerank is None:
        pagerank = pgrank(graph)
    
    # Sort the PageRank dictionary by centrality values in descending order
    sorted_pagerank = {node: pr for node, pr in sorted(pagerank.items(), key=lambda item: item[1], reverse=True)}
    top_nodes = list(sorted_pagerank.keys())[:number]  # Get the top 'number' nodes
    
    logger.info(f"Removing {number} top PageRank nodes: {top_nodes}.")
    removed_top_nodes = remove_nodes(graph=graph, nodes=top_nodes, filename="without_pgrank.csv", save_path=save_path)
    
    logger.info(f"Removed the top PageRank nodes and returned the modified graph.")
    return removed_top_nodes


def remove_top_eigenvector_ct_nodes(graph: nx.Graph, number: int, save_path: str | None = None, ev_ct: dict[str, float] | None = None) -> nx.Graph:
    '''
    Remove the top nodes by eigenvector centrality from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which to remove top eigenvector centrality nodes.
        - number (int): The number of top nodes by eigenvector centrality to remove.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - ev_ct (dict[str, float] | None, optional): Precomputed eigenvector centrality values. Default is None.

    ### Return:
        - nx.Graph: The modified graph with top eigenvector centrality nodes removed.
    '''
    if ev_ct is None:
        ev_ct = eigenvector_ct(graph)
    
    # Sort the eigenvector centrality dictionary by centrality values in descending order
    sorted_ev_ct = {node: ev for node, ev in sorted(ev_ct.items(), key=lambda item: item[1], reverse=True)}
    top_nodes = list(sorted_ev_ct.keys())[:number]  # Get the top 'number' nodes
    
    logger.info(f"Removing {number} top eigenvector centrality nodes: {top_nodes}.")
    removed_top_nodes = remove_nodes(graph=graph, nodes=top_nodes, filename="without_eigenvector_ct.csv", save_path=save_path)
    
    logger.info(f"Removed the top eigenvector centrality nodes and returned the modified graph.")
    return removed_top_nodes


def remove_top_closeness_ct_nodes(graph: nx.Graph, number: int, save_path: str | None = None, cn_ct: dict[str, float] | None = None) -> nx.Graph:
    '''
    Remove the top nodes by closeness centrality from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which to remove top closeness centrality nodes.
        - number (int): The number of top nodes by closeness centrality to remove.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - cn_ct (dict[str, float] | None, optional): Precomputed closeness centrality values. Default is None.

    ### Return:
        - nx.Graph: The modified graph with top closeness centrality nodes removed.
    '''
    if cn_ct is None:
        cn_ct = closeness_ct(graph)
    
    # Sort the closeness centrality dictionary by centrality values in descending order
    sorted_cn_ct = {node: cn for node, cn in sorted(cn_ct.items(), key=lambda item: item[1], reverse=True)}
    top_nodes = list(sorted_cn_ct.keys())[:number]  # Get the top 'number' nodes
    
    logger.info(f"Removing {number} top closeness centrality nodes: {top_nodes}.")
    removed_top_nodes = remove_nodes(graph=graph, nodes=top_nodes, filename="without_closeness_ct.csv", save_path=save_path)
    
    logger.info(f"Removed the top closeness centrality nodes and returned the modified graph.")
    return removed_top_nodes


def remove_top_node_betweennesst_ct_nodes(graph: nx.Graph, number: int, save_path: str | None = None, nd_bt_ct: dict[str, float] | None = None) -> nx.Graph:
    '''
    Remove the top nodes by node betweenness centrality from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which to remove top node betweenness centrality nodes.
        - number (int): The number of top nodes by node betweenness centrality to remove.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - nd_bt_ct (dict[str, float] | None, optional): Precomputed node betweenness centrality values. Default is None.

    ### Return:
        - nx.Graph: The modified graph with top node betweenness centrality nodes removed.
    '''
    if nd_bt_ct is None:
        nd_bt_ct = node_betweenness_ct(graph)
    
    # Sort the node betweenness centrality dictionary by centrality values in descending order
    sorted_nd_bt_ct = {node: nd_bt for node, nd_bt in sorted(nd_bt_ct.items(), key=lambda item: item[1], reverse=True)}
    top_nodes = list(sorted_nd_bt_ct.keys())[:number]  # Get the top 'number' nodes
    
    logger.info(f"Removing {number} top node betweenness centrality nodes: {top_nodes}.")
    removed_top_nodes = remove_nodes(graph=graph, nodes=top_nodes, filename="without_node_betweenness_ct.csv", save_path=save_path)
    
    logger.info(f"Removed the top node betweenness centrality nodes and returned the modified graph.")
    return removed_top_nodes


def remove_top_edge_betweenness_ct_edges(graph: nx.Graph, number: int, save_path: str | None = None, ed_bt_ct: dict[tuple[str, str], float] | None = None) -> nx.Graph:
    '''
    Remove the top edges by edge betweenness centrality from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which to remove top edge betweenness centrality edges.
        - number (int): The number of top edges by edge betweenness centrality to remove.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - ed_bt_ct (dict[tuple[str, str], float] | None, optional): Precomputed edge betweenness centrality values. Default is None.

    ### Return:
        - nx.Graph: The modified graph with top edge betweenness centrality edges removed.
    '''
    if ed_bt_ct is None:
        ed_bt_ct = edge_betweenness_ct(graph)
    
    # Sort the edge betweenness centrality dictionary by centrality values in descending order
    sorted_ed_bt_ct = {edge: ed_bt_ct[edge] for edge in sorted(ed_bt_ct, key=ed_bt_ct.get, reverse=True)}
    top_edges = list(sorted_ed_bt_ct.keys())[:number]  # Get the top 'number' edges
    
    logger.info(f"Removing {number} top edge betweenness centrality edges: {top_edges}.")
    removed_top_edges = remove_edges(graph=graph, edges=top_edges, filename="without_ed_bt_ct.csv", save_path=save_path)
    
    logger.info(f"Removed the top edge betweenness centrality edges and returned the modified graph.")
    return removed_top_edges


def remove_random_edges(graph: nx.Graph, number: int, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Remove a specified number of random edges from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which to remove random edges.
        - number (int): The number of random edges to remove.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with random edges removed.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility

    random_edges = rd.sample(list(graph.edges()), number)
    
    logger.info(f"Removing {number} random edges.")
    removed_random_edges = remove_edges(graph=graph, edges=random_edges, filename="without_rd_edges.csv", save_path=save_path)
    
    logger.info(f"Removed the random edges and returned the modified graph.")
    return removed_random_edges


def remove_edges_below_threshold(graph: nx.Graph, threshold: float, save_path: str | None = None) -> nx.Graph:
    '''
    Remove edges with weights below a specified threshold from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which to remove edges below the threshold.
        - threshold (float): The weight threshold below which edges will be removed.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - nx.Graph: The modified graph with edges below the threshold removed.
    '''
    edges_to_remove = [(u, v) for u, v, d in graph.edges(data=True) if d["weight"] < threshold]
    
    logger.info(f"Removing {len(edges_to_remove)} edges with weights below {threshold}.")
    removed_edges = remove_edges(graph=graph, edges=edges_to_remove, filename="without_below_th_edges.csv", save_path=save_path)
    
    logger.info(f"Removed the edges below the threshold and returned the modified graph.")
    return removed_edges


def remove_edges_above_threshold(graph: nx.Graph, threshold: float, save_path: str | None = None) -> nx.Graph:
    '''
    Remove edges with weights above or equal to a specified threshold from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which to remove edges above the threshold.
        - threshold (float): The weight threshold above or equal to which edges will be removed.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - nx.Graph: The modified graph with edges above the threshold removed.
    '''
    edges_to_remove = [(u, v) for u, v, d in graph.edges(data=True) if d["weight"] >= threshold]
    
    logger.info(f"Removing {len(edges_to_remove)} edges with weights above or equal to {threshold}.")
    removed_edges = remove_edges(graph=graph, edges=edges_to_remove, filename="without_above_th_edges.csv", save_path=save_path)
    
    logger.info(f"Removed the edges above the threshold and returned the modified graph.")
    return removed_edges



# ----- Increase and decrease edge weights ----- #
def increase_edge_weights(graph: nx.Graph, nodes: list[str], filename: str = "modified_graph.csv", save_path: str | None = None) -> nx.Graph:
    '''
    Increase the weights of edges connected to specified nodes, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph in which to increase edge weights.
        - nodes (list[str]): A list of nodes whose connecting edges' weights will be increased.
        - filename (str, optional): The name of the file to save the modified graph. Default is "modified_graph.csv".
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - nx.Graph: The modified graph with increased edge weights.
    '''
    modified_graph = graph.copy()  # Make a copy of the graph
    modified_edges = set()

    total_nodes = len(nodes)
    logger.info(f"Made a copy of the given graph. Will increase the weights of the edges adjacent to {total_nodes} nodes.")

    for node in nodes:
        if node in modified_graph.nodes():  # Check if node exists
            for neighbor in modified_graph.neighbors(node):
                if (node, neighbor) not in modified_edges and (neighbor, node) not in modified_edges: # Check if edge has already been increased
                    current_weight = modified_graph[node][neighbor].get("weight", avg_weight(graph))
                    new_weight = (current_weight + 1) / 2
                    modified_graph[node][neighbor]["weight"] = new_weight
                    modified_edges.add((node, neighbor))
    logger.info(f"Increased the weight of {len(modified_edges)} edges.")
    
    if save_path is not None:  # Save modified graph
        logger.info(f"A path was given, trying to save the graph to a CSV file.")
        edge_list = []
        for u, v, d in modified_graph.edges(data=True):
            edge_list.append((u, v, d["weight"]))
        df = pd.DataFrame(edge_list, columns=["source", "target", "weight"])
        save_to_csv(dataframe=df, filename=filename, save_path=save_path)
    
    logger.info(f"Returned the modified graph.")
    return modified_graph


def decrease_edge_weights(graph: nx.Graph, nodes: list[str], filename: str = "modified_graph.csv", save_path: str | None = None, factor: float = 2) -> nx.Graph:
    '''
    Decrease the weights of edges connected to specified nodes, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph in which to decrease edge weights.
        - nodes (list[str]): A list of nodes whose connecting edges' weights will be decreased.
        - filename (str, optional): The name of the file to save the modified graph. Default is "modified_graph.csv".
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - factor (float, optional): The factor by which to decrease the edge weights. Must be greater than 1. Default is 2.

    ### Return:
        - nx.Graph: The modified graph with decreased edge weights.
    '''
    if factor <= 1:
        raise ValueError("Please choose a factor bigger than 1!")
    
    modified_graph = graph.copy()  # Make a copy of the graph
    modified_edges = set()

    total_nodes = len(nodes)
    logger.info(f"Made a copy of the given graph. Will decrease the weights of the edges adjacent to {total_nodes} nodes by a factor of {factor}.")

    for node in nodes:
        if node in modified_graph.nodes():  # Check if node exists
            for neighbor in modified_graph.neighbors(node):
                if (node, neighbor) not in modified_edges and (neighbor, node) not in modified_edges:  # Check if edge has already been decreased
                    current_weight = modified_graph[node][neighbor].get("weight", avg_weight(graph))
                    new_weight = current_weight / factor
                    modified_graph[node][neighbor]["weight"] = new_weight
                    modified_edges.add((node, neighbor))
    logger.info(f"Decreased the weight of {len(modified_edges)} edges.")
    
    if save_path is not None:  # Save modified graph
        logger.info(f"A path was given, trying to save the graph to a CSV file.")
        edge_list = []
        for u, v, d in modified_graph.edges(data=True):
            edge_list.append((u, v, d["weight"]))
        df = pd.DataFrame(edge_list, columns=["source", "target", "weight"])
        save_to_csv(dataframe=df, filename=filename, save_path=save_path)
    
    logger.info(f"Returned the modified graph.")
    return modified_graph


def increase_random_edge_weights(graph: nx.Graph, number: int = 10, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Increase the weights of edges connected to a specified number of random nodes, save the modified graph to a CSV file if a path is provided, and return the modified graph.

    ### Args:
        - graph (nx.Graph): The graph in which to increase edge weights.
        - number (int, optional): The number of random nodes whose connecting edges' weights will be increased. Default is 10.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with increased edge weights.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility

    random_nodes = rd.sample(list(graph.nodes()), number)
    logger.info(f"Increasing the weights of the edges adjacent to {number} nodes: {random_nodes}.")
    random_nodes_increased = increase_edge_weights(graph=graph, nodes=random_nodes, filename="rd_edges_increased.csv", save_path=save_path)
    
    logger.info(f"Increased the edge weights and returned the modified graph.")
    return random_nodes_increased


def decrease_random_edge_weights(graph: nx.Graph, number: int = 10, factor: float = 2, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Decrease the weights of edges connected to a specified number of random nodes, save the modified graph to a CSV file if a path is provided, and return the modified graph.

    ### Args:
        - graph (nx.Graph): The graph in which to decrease edge weights.
        - number (int, optional): The number of random nodes whose connecting edges' weights will be decreased. Default is 10.
        - factor (float, optional): The factor by which to decrease the edge weights. Must be greater than 1. Default is 2.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with decreased edge weights.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility

    random_nodes = rd.sample(list(graph.nodes()), number)
    logger.info(f"Decreasing the weights of the edges adjacent to {number} nodes: {random_nodes}.")
    random_nodes_decreased = decrease_edge_weights(graph=graph, nodes=random_nodes, filename="rd_edges_decreased.csv", save_path=save_path, factor=factor)
    
    logger.info(f"Decreased the edge weights and returned the modified graph.")
    return random_nodes_decreased



# ----- Add edges ----- #
def add_edges(graph: nx.Graph, edges: list[tuple[str, str]], filename: str = "modified_graph.csv", save_path: str | None = None, weight: float | None = None) -> nx.Graph:
    '''
    Add specified edges to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - edges (list[tuple[str, str]]): A list of edges to be added.
        - filename (str, optional): The name of the file to save the modified graph. Default is "modified_graph.csv".
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - weight (float | None, optional): The weight of the edges to be added. If None, use the average weight of the graph. Default is None.

    ### Return:
        - nx.Graph: The modified graph with specified edges added.
    '''
    modified_graph = graph.copy()  # Make a copy of the graph
    
    if weight is None:
        weight = avg_weight(graph)

    logger.info(f"Adding {len(edges)} edges with weight {weight}.")

    for u, v in edges:
        if not modified_graph.has_edge(u, v):
            modified_graph.add_edge(u, v, weight=weight)  # Add the relevant edges
    
    if save_path is not None: # Save modified graph
        logger.info(f"A path was given, trying to save the graph to a CSV file.")
        edge_list = []
        for u, v, d in modified_graph.edges(data=True):
            edge_list.append((u, v, d["weight"]))
        df = pd.DataFrame(edge_list, columns=["source", "target", "weight"])
        save_to_csv(dataframe=df, filename=filename, save_path=save_path)

    logger.info(f"Returned the modified graph.")
    return modified_graph


def add_community_edges(graph: nx.Graph, number: int = 10, save_path: str | None = None, communities: list[list[str]] | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of edges between random nodes from different communities to a copy of the graph, optionally save the modified graph to a CSV file, and return it.
    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to be added between communities. Default is 10.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - communities (list[list[str]] | None, optional): Precomputed communities. If None, communities will be computed. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with edges added between communities.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if communities is None:
        communities = communities(graph)
    
    added_community_edges = []
    for _ in range(number):
        random_communities = rd.sample(communities, 2)  # Select two random communities
        # Select random nodes from each community
        u = rd.choice(list(random_communities[0]))
        v = rd.choice(list(random_communities[1]))
        added_community_edges.append((u, v))

    logger.info(f"Adding {number} edges between random nodes from different communities: {added_community_edges}.")
    added_edges = add_edges(graph=graph, edges=added_community_edges, filename="with_community_edges.csv", save_path=save_path)
    
    logger.info(f"Added the community edges and returned the modified graph.")
    return added_edges


def add_random_edges(graph: nx.Graph, number: int = 100, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of random edges to add. Default is 100.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with random edges added.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    random_nodes = rd.sample(list(graph.nodes()), number * 2)
    random_edges = [(random_nodes[i], random_nodes[i + 1]) for i in range(0, len(random_nodes), 2)]
    
    logger.info(f"Adding {number} random edges: {random_edges}.")
    added_random_edges = add_edges(graph=graph, edges=random_edges, filename="with_rd_edges.csv", save_path=save_path)
    
    logger.info(f"Added the random edges and returned the modified graph.")
    return added_random_edges


def add_rd_degree_ct_edges(graph: nx.Graph, number: int = 4, centrality_threshold: float = 0.3, centrality: dict[str, float] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between nodes with a high and low degree centrality to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - centrality_threshold (float, optional): The degree centrality threshold to distinguish high and low values. Default is 0.3.
        - centrality (dict[str, float] | None, optional): Precomputed degree centrality values. If None, centrality will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between central and peripheral nodes.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if centrality is None:
        centrality = degree_ct(graph)
    
    central_nodes = []
    peripheral_nodes = []
    
    for node, centrality_value in sorted(centrality.items(), key=lambda item: item[1], reverse=True):
        if centrality_value >= centrality_threshold:
            central_nodes.append(node)
        else:
            peripheral_nodes.append(node)

    try:
        to_add = rd.sample([(u, v) for u in central_nodes for v in peripheral_nodes], number)
    except ValueError:
        raise ValueError(f"There are not enough nodes either below or above the threshold. Please choose a different threshold than {centrality_threshold} or a lower number than {number}.")
    
    logger.info(f"Adding {number} random edges between nodes with a high and low degree centrality: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_rd_degree_ct_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_rd_eigenvector_ct_edges(graph: nx.Graph, number: int = 4, centrality_threshold: float = 0.3, centrality: dict[str, float] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between nodes with a high and low eigenvector centrality to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - centrality_threshold (float, optional): The eigenvector centrality threshold to distinguish high and low values. Default is 0.3.
        - centrality (dict[str, float] | None, optional): Precomputed eigenvector centrality values. If None, centrality will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between central and peripheral nodes.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if centrality is None:
        centrality = eigenvector_ct(graph)
    
    central_nodes = []
    peripheral_nodes = []
    
    for node, centrality_value in sorted(centrality.items(), key=lambda item: item[1], reverse=True):
        if centrality_value >= centrality_threshold:
            central_nodes.append(node)
        else:
            peripheral_nodes.append(node)

    try:
        to_add = rd.sample([(u, v) for u in central_nodes for v in peripheral_nodes], number)
    except ValueError:
        raise ValueError(f"There are not enough nodes either below or above the threshold. Please choose a different threshold than {centrality_threshold} or a lower number than {number}.")
    
    logger.info(f"Adding {number} random edges between nodes with a high and low eigenvector centrality: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_rd_eigenvector_ct_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_rd_closeness_ct_edges(graph: nx.Graph, number: int = 4, centrality_threshold: float = 0.3, centrality: dict[str, float] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between nodes with a high and low closeness centrality to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - centrality_threshold (float, optional): The closeness centrality threshold to distinguish high and low values. Default is 0.3.
        - centrality (dict[str, float] | None, optional): Precomputed closeness centrality values. If None, centrality will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between central and peripheral nodes.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if centrality is None:
        centrality = closeness_ct(graph)
    
    central_nodes = []
    peripheral_nodes = []
    
    for node, centrality_value in sorted(centrality.items(), key=lambda item: item[1], reverse=True):
        if centrality_value >= centrality_threshold:
            central_nodes.append(node)
        else:
            peripheral_nodes.append(node)

    try:
        to_add = rd.sample([(u, v) for u in central_nodes for v in peripheral_nodes], number)
    except ValueError:
        raise ValueError(f"There are not enough nodes either below or above the threshold. Please choose a different threshold than {centrality_threshold} or a lower number than {number}.")
    
    logger.info(f"Adding {number} random edges between nodes with a high and low closeness centrality: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_rd_cn_ct_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_rd_pgrank_edges(graph: nx.Graph, number: int = 4, centrality_threshold: float = 0.08, centrality: dict[str, float] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between nodes with a high and low PageRank to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - centrality_threshold (float, optional): The PageRank centrality threshold to distinguish high and low values. Default is 0.08.
        - centrality (dict[str, float] | None, optional): Precomputed PageRank centrality values. If None, centrality will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between central and peripheral nodes.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if centrality is None:
        centrality = pgrank(graph)
    
    central_nodes = []
    peripheral_nodes = []
    
    for node, centrality_value in sorted(centrality.items(), key=lambda item: item[1], reverse=True):
        if centrality_value >= centrality_threshold:
            central_nodes.append(node)
        else:
            peripheral_nodes.append(node)

    try:
        to_add = rd.sample([(u, v) for u in central_nodes for v in peripheral_nodes], number)
    except ValueError:
        raise ValueError(f"There are not enough nodes either below or above the threshold. Please choose a different threshold than {centrality_threshold} or a lower number than {number}.")
    
    logger.info(f"Adding {number} random edges between nodes with a high and low PageRank: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_rd_pgrank_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_rd_node_betweenness_ct_edges(graph: nx.Graph, number: int = 4, centrality_threshold: float = 0.3, centrality: dict[str, float] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between nodes with a high and low node betweenness centrality to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - centrality_threshold (float, optional): The node betweenness centrality threshold to distinguish high and low values. Default is 0.3.
        - centrality (dict[str, float] | None, optional): Precomputed node betweenness centrality values. If None, centrality will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between nodes with a high and low node betweenness centrality.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if centrality is None:
        centrality = node_betweenness_ct(graph)
        
    central_nodes = []
    peripheral_nodes = []
    
    for node, centrality_value in sorted(centrality.items(), key=lambda item: item[1], reverse=True):
        if centrality_value >= centrality_threshold:
            central_nodes.append(node)
        else:
            peripheral_nodes.append(node)

    try:
        to_add = rd.sample([(u, v) for u in central_nodes for v in peripheral_nodes], number)
    except ValueError:
        raise ValueError(f"There are not enough nodes either below or above the threshold. Please choose a different threshold than {centrality_threshold} or a lower number than {number}.")
    
    logger.info(f"Adding {number} random edges between nodes with a high and low node betweenness centrality: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_rd_node_betweenness_ct_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_rd_center_edges(graph: nx.Graph, number: int = 4, center: list[list[str]] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between center nodes and non-center nodes to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - center (list[list[str]] | None, optional): Precomputed center nodes. If None, center will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between center and non-center nodes.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if center is None:
        center = find_center(graph)
    
    center_nodes = set(node for subgroup in center for node in subgroup) # Put all given nodes in a single set
    all_nodes = set(graph.nodes())
    peripheral_nodes = list(all_nodes - center_nodes)

    to_add = rd.sample([(u, v) for u in center_nodes for v in peripheral_nodes], number)
    
    logger.info(f"Adding {number} random edges between center and non-center nodes: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_rd_center_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_rd_barycenter_edges(graph: nx.Graph, number: int = 4, center: list[list[str]] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between barycenter nodes and non-barycenter nodes to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - center (list[list[str]] | None, optional): Precomputed barycenter nodes. If None, barycenter will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between barycenter and non-barycenter nodes.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if center is None:
        center = find_barycenter(graph)
    
    center_nodes = set(node for subgroup in center for node in subgroup) # Put all given nodes in a single set
    all_nodes = set(graph.nodes())
    peripheral_nodes = list(all_nodes - center_nodes)

    to_add = rd.sample([(u, v) for u in center_nodes for v in peripheral_nodes], number)
    
    logger.info(f"Adding {number} random edges between barycenter and non-barycenter nodes: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_rd_barycenter_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_rd_periphery_edges(graph: nx.Graph, number: int = 4, peripheries: list[list[str]] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between periphery nodes and non-periphery nodes to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - peripheries (list[list[str]] | None, optional): Precomputed periphery nodes. If None, periphery will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between periphery and non-periphery nodes.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if peripheries is None:
        peripheries = periphery(graph)
    
    periphery_nodes = set(node for subgroup in peripheries for node in subgroup) # Put all given nodes in a single set
    all_nodes = set(graph.nodes())
    non_peripheral_nodes = list(all_nodes - periphery_nodes)

    to_add = rd.sample([(u, v) for u in periphery_nodes for v in non_peripheral_nodes], number)
    
    logger.info(f"Adding {number} random edges between periphery and non-periphery nodes: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_rd_periphery_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_low_degree_ct_edges(graph: nx.Graph, number: int = 4, centrality_threshold: float = 0.3, centrality: dict[str, float] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between nodes with a low degree centrality to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - centrality_threshold (float, optional): The degree centrality threshold to distinguish low values. Default is 0.3.
        - centrality (dict[str, float] | None, optional): Precomputed degree centrality values. If None, centrality will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between nodes with a low degree centrality.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if centrality is None:
        centrality = degree_ct(graph)
    
    low_central_nodes = []
    
    for node, centrality_value in sorted(centrality.items(), key=lambda item: item[1]):
        if centrality_value < centrality_threshold:
            low_central_nodes.append(node)
    
    # Select pairs of low central nodes
    try:
        logger.info(f"Trying to select {number} pairs of nodes.")
        to_add = rd.sample([(u, v) for u in low_central_nodes for v in low_central_nodes], number)
    except ValueError:
        logger.error(f"Could not select enough pairs.")
        raise ValueError(f"There are not enough nodes with a low centrality. Please choose a higher threshold than {centrality_threshold} or a lower number than {number}.")
    
    logger.info(f"Adding {number} random edges between nodes with a low degree centrality: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_low_degree_ct_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_low_eigenvector_ct_edges(graph: nx.Graph, number: int = 4, centrality_threshold: float = 0.3, centrality: dict[str, float] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between nodes with a low eigenvector centrality to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - centrality_threshold (float, optional): The eigenvector centrality threshold to distinguish low values. Default is 0.3.
        - centrality (dict[str, float] | None, optional): Precomputed eigenvector centrality values. If None, centrality will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between nodes with a low eigenvector centrality.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if centrality is None:
        centrality = eigenvector_ct(graph)
    
    low_central_nodes = []
    
    for node, centrality_value in sorted(centrality.items(), key=lambda item: item[1]):
        if centrality_value < centrality_threshold:
            low_central_nodes.append(node)
    
    # Select pairs of low central nodes
    try:
        logger.info(f"Trying to select {number} pairs of nodes.")
        to_add = rd.sample([(u, v) for u in low_central_nodes for v in low_central_nodes], number)
    except ValueError:
        logger.error(f"Could not select enough pairs.")
        raise ValueError(f"There are not enough nodes with a low centrality. Please choose a higher threshold than {centrality_threshold} or a lower number than {number}.")
    
    logger.info(f"Adding {number} random edges between nodes with a low eigenvector centrality: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_low_eigenvector_ct_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_low_closeness_ct_edges(graph: nx.Graph, number: int = 4, centrality_threshold: float = 0.3, centrality: dict[str, float] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between nodes with a low closeness centrality to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - centrality_threshold (float, optional): The closeness centrality threshold to distinguish low values. Default is 0.3.
        - centrality (dict[str, float] | None, optional): Precomputed closeness centrality values. If None, centrality will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between nodes with a low closeness centrality.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if centrality is None:
        centrality = closeness_ct(graph)
    
    low_central_nodes = []
    
    for node, centrality_value in sorted(centrality.items(), key=lambda item: item[1]):
        if centrality_value < centrality_threshold:
            low_central_nodes.append(node)

    # Select pairs of low central nodes
    try:
        logger.info(f"Trying to select {number} pairs of nodes.")
        to_add = rd.sample([(u, v) for u in low_central_nodes for v in low_central_nodes], number)
    except ValueError:
        logger.error(f"Could not select enough pairs.")
        raise ValueError(f"There are not enough nodes with a low centrality. Please choose a higher threshold than {centrality_threshold} or a lower number than {number}.")
    
    logger.info(f"Adding {number} random edges between nodes with a low closeness centrality: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_low_closeness_ct_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_low_pgrank_edges(graph: nx.Graph, number: int = 4, centrality_threshold: float = 0.3, centrality: dict[str, float] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between nodes with a low PageRank to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - centrality_threshold (float, optional): The PageRank threshold to distinguish low values. Default is 0.3.
        - centrality (dict[str, float] | None, optional): Precomputed PageRank values. If None, centrality will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between nodes with a low PageRank.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if centrality is None:
        centrality = pgrank(graph)
    
    low_central_nodes = []
    
    for node, centrality_value in sorted(centrality.items(), key=lambda item: item[1]):
        if centrality_value < centrality_threshold:
            low_central_nodes.append(node)
    
    # Select pairs of low central nodes
    try:
        logger.info(f"Trying to select {number} pairs of nodes.")
        to_add = rd.sample([(u, v) for u in low_central_nodes for v in low_central_nodes], number)
    except ValueError:
        logger.error(f"Could not select enough pairs.")
        raise ValueError(f"There are not enough nodes with a low PageRank. Please choose a higher threshold than {centrality_threshold} or a lower number than {number}.")
    
    logger.info(f"Adding {number} random edges between nodes with a low PageRank: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_low_pgrank_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_low_node_betweenness_ct_edges(graph: nx.Graph, number: int = 4, centrality_threshold: float = 0.3, centrality: dict[str, float] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between nodes with a low node betweenness centrality to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - centrality_threshold (float, optional): The node betweenness centrality threshold to distinguish low values. Default is 0.3.
        - centrality (dict[str, float] | None, optional): Precomputed node betweenness centrality values. If None, centrality will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between nodes with a low node betweenness centrality.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if centrality is None:
        centrality = node_betweenness_ct(graph)
    
    low_central_nodes = []
    
    for node, centrality_value in sorted(centrality.items(), key=lambda item: item[1]):
        if centrality_value < centrality_threshold:
            low_central_nodes.append(node)

    # Select pairs of low central nodes
    try:
        logger.info(f"Trying to select {number} pairs of nodes.")
        to_add = rd.sample([(u, v) for u in low_central_nodes for v in low_central_nodes], number)
    except ValueError:
        logger.error(f"Could not select enough pairs.")
        raise ValueError(f"There are not enough nodes with a low centrality. Please choose a higher threshold than {centrality_threshold} or a lower number than {number}.")
    
    logger.info(f"Adding {number} random edges between nodes with low node betweenness centrality: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_low_node_betweenness_ct_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_pc_degree_ct_edges(graph: nx.Graph, number: int = 10, percent: int = 50, centrality: dict[str, float] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between the top percentage and bottom percentage of nodes based on degree centrality to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 10.
        - percent (int, optional): The percentage of top and bottom nodes to consider based on degree centrality. Default is 50.
        - centrality (dict[str, float] | None, optional): Precomputed degree centrality values. If None, centrality will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between the top percentage and bottom percentage of nodes based on degree centrality.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if centrality is None:
        centrality = degree_ct(graph)

    total_nodes = len(graph.nodes())
    num_nodes = int(percent * total_nodes / 100) # The actual number of nodes to sample from the top and the bottom
    logger.info(f"The actual number of nodes to be sampled from each the top and the bottom is {num_nodes}.")
    to_add = []

    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    top_nodes = [node for node, _ in sorted_nodes[:num_nodes]]
    bottom_nodes = [node for node, _ in sorted_nodes[-num_nodes:]]
    
    for _ in range(number):
        u = rd.choice(top_nodes)
        v = rd.choice(bottom_nodes)
    
        to_add.append((u, v))
    
    logger.info(f"Adding {number} random edges between the top and bottom {percent}% of nodes based on degree centrality: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_pc_degree_ct_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_pc_eigenvector_ct_edges(graph: nx.Graph, number: int = 4, percent: int = 50, centrality: dict[str, float] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between the top percentage and bottom percentage of nodes based on eigenvector centrality to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - percent (int, optional): The percentage of top and bottom nodes to consider based on eigenvector centrality. Default is 50.
        - centrality (dict[str, float] | None, optional): Precomputed eigenvector centrality values. If None, centrality will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between the top percentage and bottom percentage of nodes based on eigenvector centrality.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if centrality is None:
        centrality = eigenvector_ct(graph)

    total_nodes = len(graph.nodes())
    num_nodes = int(percent * total_nodes / 100)  # The actual number of nodes to sample from the top and the bottom
    logger.info(f"The actual number of nodes to be sampled from each the top and the bottom is {num_nodes}.")
    to_add = []

    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    top_nodes = [node for node, _ in sorted_nodes[:num_nodes]]
    bottom_nodes = [node for node, _ in sorted_nodes[-num_nodes:]]
    
    for _ in range(number):
        u = rd.choice(top_nodes)
        v = rd.choice(bottom_nodes)
    
        to_add.append((u, v))
    
    logger.info(f"Adding {number} random edges between the top and bottom {percent}% of nodes based on eigenvector centrality: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_pc_eigenvector_ct_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_pc_closeness_ct_edges(graph: nx.Graph, number: int = 4, percent: int = 50, centrality: dict[str, float] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between the top percentage and bottom percentage of nodes based on closeness centrality to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - percent (int, optional): The percentage of top and bottom nodes to consider based on closeness centrality. Default is 50.
        - centrality (dict[str, float] | None, optional): Precomputed closeness centrality values. If None, centrality will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between the top percentage and bottom percentage of nodes based on closeness centrality.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if centrality is None:
        centrality = closeness_ct(graph)

    total_nodes = len(graph.nodes())
    num_nodes = int(percent * total_nodes / 100)  # The actual number of nodes to sample from the top and the bottom
    logger.info(f"The actual number of nodes to be sampled from each the top and the bottom is {num_nodes}.")
    to_add = []

    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    top_nodes = [node for node, _ in sorted_nodes[:num_nodes]]
    bottom_nodes = [node for node, _ in sorted_nodes[-num_nodes:]]
    
    for _ in range(number):
        u = rd.choice(top_nodes)
        v = rd.choice(bottom_nodes)
    
        to_add.append((u, v))
    
    logger.info(f"Adding {number} random edges between the top and bottom {percent}% of nodes based on closeness centrality: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_pc_closeness_ct_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_pc_pgrank_edges(graph: nx.Graph, number: int = 4, percent: int = 50, centrality: dict[str, float] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between the top percentage and bottom percentage of nodes based on PageRank to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - percent (int, optional): The percentage of top and bottom nodes to consider based on PageRank. Default is 50.
        - centrality (dict[str, float] | None, optional): Precomputed PageRank values. If None, centrality will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between the top percentage and bottom percentage of nodes based on PageRank.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if centrality is None:
        centrality = pgrank(graph)

    total_nodes = len(graph.nodes())
    num_nodes = int(percent * total_nodes / 100)  # The actual number of nodes to sample from the top and the bottom
    logger.info(f"The actual number of nodes to be sampled from each the top and the bottom is {num_nodes}.")
    to_add = []

    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    top_nodes = [node for node, _ in sorted_nodes[:num_nodes]]
    bottom_nodes = [node for node, _ in sorted_nodes[-num_nodes:]]
    
    for _ in range(number):
        u = rd.choice(top_nodes)
        v = rd.choice(bottom_nodes)
    
        to_add.append((u, v))
    
    logger.info(f"Adding {number} random edges between the top and bottom {percent}% of nodes based on PageRank: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_pc_pgrank_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_pc_node_betweenness_ct_edges(graph: nx.Graph, number: int = 4, percent: int = 50, centrality: dict[str, float] | None = None, save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Add a specified number of random edges between the top percentage and bottom percentage of nodes based on node betweenness centrality to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - number (int, optional): The number of edges to add. Default is 4.
        - percent (int, optional): The percentage of top and bottom nodes to consider based on node betweenness centrality. Default is 50.
        - centrality (dict[str, float] | None, optional): Precomputed node betweenness centrality values. If None, centrality will be computed. Default is None.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with added edges between the top percentage and bottom percentage of nodes based on node betweenness centrality.
    '''
    if seed is not None:
        rd.seed(seed)  # Set the seed for reproducibility
    
    if centrality is None:
        centrality = node_betweenness_ct(graph)

    total_nodes = len(graph.nodes())
    num_nodes = int(percent * total_nodes / 100)  # The actual number of nodes to sample from the top and the bottom
    logger.info(f"The actual number of nodes to be sampled from each the top and the bottom is {num_nodes}.")
    to_add = []

    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    top_nodes = [node for node, _ in sorted_nodes[:num_nodes]]
    bottom_nodes = [node for node, _ in sorted_nodes[-num_nodes:]]
    
    for _ in range(number):
        u = rd.choice(top_nodes)
        v = rd.choice(bottom_nodes)
    
        to_add.append((u, v))
    
    logger.info(f"Adding {number} random edges between the top and bottom {percent}% of nodes based on node betweenness centrality: {to_add}.")
    added_edges = add_edges(graph=graph, edges=to_add, filename="with_pc_nd_bt_ct_edges.csv", save_path=save_path)
    
    logger.info(f"Added the edges and returned the modified graph.")
    return added_edges


def add_specific_edges(graph: nx.Graph, edge_list: list[tuple[str, str, float]], save_path: str | None = None) -> nx.Graph:
    '''
    Add specified edges with given weights to a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph to which edges will be added.
        - edge_list (list[tuple[str, str, float]]): A list of tuples, where each tuple contains two nodes and a weight (u, v, weight).
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - nx.Graph: The modified graph with the specified edges added.
    '''
    modified_graph = graph.copy()
    
    logger.info(f"Trying to add {len(edge_list)} edges.")
    for u, v, weight in edge_list:
        if not modified_graph.has_edge(u, v):
            modified_graph.add_edge(u, v, weight=weight)  # Add the relevant edge
    
    if save_path is not None:  # Save modified graph
        logger.info(f"A path was given, trying to save the graph to a CSV file.")
        edge_list = []
        for u, v, d in modified_graph.edges(data=True):
            edge_list.append((u, v, d["weight"]))
        df = pd.DataFrame(edge_list, columns=["source", "target", "weight"])
        save_to_csv(dataframe=df, filename="with_specific_edges.csv", save_path=save_path)
    
    logger.info(f"Added the specified edges and returned the modified graph.")
    return modified_graph



# ----- Remove edges ----- #
def remove_edges(graph: nx.Graph, edges: list[tuple[str, str]], filename: str = "modified_graph.csv", save_path: str | None = None) -> nx.Graph:
    '''
    Remove specified edges from a copy of the graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph from which edges will be removed.
        - edges (list[tuple[str, str]]): A list of edges to be removed.
        - filename (str, optional): The name of the file to save the modified graph. Default is "modified_graph.csv".
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - nx.Graph: The modified graph with specified edges removed.
    '''
    modified_graph = graph.copy()  # Make a copy of the graph

    logger.info(f"Removing {len(edges)} edges.")

    for u, v in edges:
        if modified_graph.has_edge(u, v):
            modified_graph.remove_edge(u, v)  # Remove the relevant edges
    
    if save_path is not None: # Save modified graph
        logger.info(f"A path was given, trying to save the graph to a CSV file.")
        edge_list = []
        for u, v, d in modified_graph.edges(data=True):
            edge_list.append((u, v, d["weight"]))
        df = pd.DataFrame(edge_list, columns=["source", "target", "weight"])
        save_to_csv(dataframe=df, filename=filename, save_path=save_path)

    logger.info(f"Returned the modified graph.")
    return modified_graph



# ----- Edge Swapping ----- #
def double_edge_swap_subgraphs(graph: nx.Graph, nswap: int = 1, max_tries: int = 100, filename: str = "modified_graph.csv", save_path: str | None = None, seed: int | float | str | bytes | bytearray | None = None) -> nx.Graph:
    '''
    Perform a double-edge swap on each subgraph of the given graph, optionally save the modified graph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The graph on which to perform double-edge swaps.
        - nswap (int, optional): The number of edge swaps to perform. Default is 1.
        - max_tries (int, optional): The maximum number of attempts to swap edges. Default is 100.
        - filename (str, optional): The name of the file to save the modified graph. Default is 'modified_graph.csv'.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.
        - seed (int | float | str | bytes | bytearray | None, optional): The seed for the random number generator. Default is None.

    ### Return:
        - nx.Graph: The modified graph with double-edge swaps performed on each subgraph.
    '''
    swapped_graph = graph.copy()  # Make a copy of the graph

    # Iterate over subgraphs
    logger.info(f"Going through all subgraphs and trying to perform a double edge swap on them.")
    index = 0
    for subgraph in subgraphs(swapped_graph):
        subgraph_copy = subgraph.copy()
        
        try:  # Perform edge swaps on the subgraph
            nx.double_edge_swap(subgraph_copy, nswap=nswap, max_tries=max_tries, seed=seed)
        except nx.NetworkXError as e:
            logger.warning(f"Edge swap failed for subgraph {index}. Error: {e}")
            continue
    
        swapped_graph.update(subgraph_copy)  # Update the main graph with the changes
        
        # Remove old edges that were swapped out
        old_edges = set(subgraph.edges) - set(subgraph_copy.edges)
        for u, v in old_edges:
            if swapped_graph.has_edge(u, v):
                swapped_graph.remove_edge(u, v)

        index += 1

    if save_path is not None:  # Save modified graph
        logger.info(f"A path was given, trying to save the graph to a CSV file.")
        edge_list = []
        for u, v, d in swapped_graph.edges(data=True):
            edge_list.append((u, v, d["weight"]))
        df = pd.DataFrame(edge_list, columns=["source", "target", "weight"])
        save_to_csv(dataframe=df, filename=filename, save_path=save_path)

    logger.info(f"Returned the modified graph.")
    return swapped_graph



# ----- Create subgraph + environment ----- #
def create_subgraph_with_neighbors(graph: nx.Graph, nodes: list[str], filename: str = "subgraph.csv", save_path: str | None = None) -> nx.Graph:
    '''
    Create a subgraph containing the specified disease nodes and their first-degree neighbors, optionally save the subgraph to a CSV file, and return it.

    ### Args:
        - graph (nx.Graph): The original graph from which to create the subgraph.
        - nodes (list[str]): A list of nodes to include in the subgraph along with their first-degree neighbors.
        - filename (str, optional): The name of the file to save the subgraph. Default is 'subgraph.csv'.
        - save_path (str | None, optional): The directory path where the CSV file will be saved. Default is None.

    ### Return:
        - nx.Graph: The subgraph containing the specified nodes and their first-degree neighbors.
    '''
    nodes_to_include = set()

    for node in nodes:
        if node in graph:
            nodes_to_include.add(node)  # Add the node itself
            neighbors = list(graph.neighbors(node))
            nodes_to_include.update(neighbors)  # Add first-degree neighbors

    subgraph = graph.subgraph(nodes_to_include).copy()  # Create the subgraph

    if save_path is not None:  # Save subgraph
        logger.info(f"A path was given, trying to save the subgraph to a CSV file.")
        edge_list = []
        for u, v, d in subgraph.edges(data=True):
            edge_list.append((u, v, d["weight"]))
        df = pd.DataFrame(edge_list, columns=["source", "target", "weight"])
        save_to_csv(dataframe=df, filename=filename, save_path=save_path)

    logger.info(f"Returned the subgraph containing the specified nodes and their first-degree neighbors.")
    return subgraph