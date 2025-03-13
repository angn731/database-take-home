#!/usr/bin/env python3
import json
import os
import sys
import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

# Add project root to path to import scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

# Import constants
from scripts.constants import (
    NUM_NODES,
    MAX_EDGES_PER_NODE,
    MAX_TOTAL_EDGES,
)

MAX_WEIGHT = 10

def load_graph(graph_file):
    """Load graph from a JSON file."""
    with open(graph_file, "r") as f:
        return json.load(f)


def load_results(results_file):
    """Load query results from a JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)

def load_queries(queries_file):
    """Load queries from the given JSON file."""
    with open(queries_file, "r") as f:
        queries = json.load(f)
    return queries


def save_graph(graph, output_file):
    """Save graph to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(graph, f, indent=2)


def verify_constraints(graph, max_edges_per_node, max_total_edges):
    """Verify that the graph meets all constraints."""
    # Check total edges
    total_edges = sum(len(edges) for edges in graph.values())
    if total_edges > max_total_edges:
        print(
            f"WARNING: Graph has {total_edges} edges, exceeding limit of {max_total_edges}"
        )
        return False

    # Check max edges per node
    max_node_edges = max(len(edges) for edges in graph.values())
    if max_node_edges > max_edges_per_node:
        print(
            f"WARNING: A node has {max_node_edges} edges, exceeding limit of {max_edges_per_node}"
        )
        return False

    # Check all nodes are present
    if len(graph) != NUM_NODES:
        print(f"WARNING: Graph has {len(graph)} nodes, should have {NUM_NODES}")
        return False

    # Check edge weights are valid (between 0 and 10)
    for node, edges in graph.items():
        for target, weight in edges.items():
            if weight <= 0 or weight > 10:
                print(f"WARNING: Edge {node} -> {target} has invalid weight {weight}")
                return False

    return True


def query_frequencies():
    """
    Helper function that returns a dictionary mapping each query to 
    the frequency with which it is queried. 
    """
    queries_file = os.path.join(project_dir, "data", "queries.json")
    queries = load_queries(queries_file)

    frequency_map = {}

    for query_node in queries:
        frequency_map.setdefault(query_node, 0)
        frequency_map[query_node] += 1
    
    return frequency_map


def prune_edges(optimized_graph, query_frequencies_dict, max_total_edges=1000, max_edges_per_node=MAX_EDGES_PER_NODE):
    """
    Prune edges to ensure the graph stays within the edge limit, prioritizing edges connected to frequently queried nodes.

    Args:
        optimized_graph: Initial graph represented as an adjacency list
        query_frequencies_dict: Dictionary mapping each node to its query frequency
        max_total_edges: Maximum allowed total number of edges in the graph
        max_edges_per_node: Maximum allowed edges per node

    Returns:
        Pruned graph with edges removed
    """
    total_edges = sum(len(edges) for edges in optimized_graph.values())

    # initialize a dictionary storing the (sum of edge weights)/(# edges) for each node
    weights_to_edges_ratios = {}

    if total_edges > max_total_edges:
        total_edges_to_remove = total_edges - max_total_edges
        print(f"Initial graph has {total_edges} edges, need to remove {total_edges_to_remove} edges.")

        # Go through each node and prune edges if necessary
        for node, edges in optimized_graph.items():
            # Sort neighbors by edge weight (previously reweighted graph based on query frequency)
            sorted_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True) # descending order

            # Remove edges that lead to nodes with low query frequency
            edges_to_remove_from_node = len(edges) - max_edges_per_node

            # Prune the edges of the node if necessary
            if edges_to_remove_from_node > 0:
                for i in range(edges_to_remove_from_node):
                    neighbor, _ = sorted_edges[i]
                    total_edges_to_remove -= 1
                    print(f'pruning edges of node {node} and trying to remove {neighbor}')
                    del optimized_graph[node][neighbor]
                    # print(f"Removed edge from node {node} to {neighbor}")

            sum_of_edge_weights = sum(weight for _, weight in optimized_graph[node].items())
            num_edges = len(optimized_graph[node])
            if num_edges > 0:
                weights_to_edges_ratios[node] = sum_of_edge_weights / num_edges
            else:
                weights_to_edges_ratios[node] = 0 
        
        if total_edges_to_remove > 0:
            sorted_nodes_by_ratio = sorted(weights_to_edges_ratios.items(), key=lambda x: x[1]) # tuples of the form (node, ratio), ascending order

            for node, ratio in sorted_nodes_by_ratio:
                edges = optimized_graph[node]

                if total_edges_to_remove == 0:
                    break

                # skip nodes with only one edge
                if len(optimized_graph[node]) <= 1:
                    continue

                # otherwise, remove the least important edge
                sorted_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)
                neighbor, _ = sorted_edges[-1] 
                # print(f'pruning edges of node {node} and trying to remove {neighbor}')
                del optimized_graph[node][neighbor]

                total_edges_to_remove -= 1
                # print(f"Removed edge from node {node} to {neighbor}")

    return optimized_graph


def bfs_shortest_path(graph, start_node, targeted_nodes):
    """
    Perform BFS to find the shortest path from the start_node to any of the targeted nodes.

    Args:
        graph (dict): The graph represented as an adjacency list.
        start_node (int): The starting node for BFS.
        targeted_nodes (set): Set of nodes that are targeted.

    Returns:
        int: The shortest distance to the closest targeted node, or float('inf') if no path exists.
    """
    # Convert integers in targeted_nodes to strings
    targeted_nodes = set(str(node) for node in targeted_nodes)

    # Initialize distances as infinity
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    
    # BFS queue
    queue = [start_node]
    
    while queue:
        current_node = queue.pop(0)
        # If we reach a targeted node, return the distance
        if current_node in targeted_nodes:
            # print('i am a targeted node')
            return distances[current_node]
        
        # Explore neighbors
        for neighbor in graph[current_node]:
            if distances[neighbor] == float('inf'):  # Not visited yet
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    
    # No path found to target node
    return float('inf')


def optimize_graph(
    initial_graph,
    results,
    num_nodes=NUM_NODES,
    max_total_edges=int(MAX_TOTAL_EDGES),
    max_edges_per_node=MAX_EDGES_PER_NODE,
):
    """
    Optimize the graph to improve random walk query performance.

    Args:
        initial_graph: Initial graph adjacency list
        results: Results from queries on the initial graph
        num_nodes: Number of nodes in the graph
        max_total_edges: Maximum total edges allowed
        max_edges_per_node: Maximum edges per node

    Returns:
        Optimized graph
    """
    print("Starting graph optimization...")

    # Create a copy of the initial graph to modify
    optimized_graph = {}
    for node, edges in initial_graph.items():
        optimized_graph[node] = dict(edges)

    # =============================================================
    # TODO: Implement your optimization strategy here
    # =============================================================
    #
    # Your goal is to optimize the graph structure to:
    # 1. Increase the success rate of queries
    # 2. Minimize the path length for successful queries
    #
    # You have access to:
    # - initial_graph: The current graph structure
    # - results: The results of running queries on the initial graph
    #
    # Query results contain:
    # - Each query's target node
    # - Whether the query was successful
    # - The path taken during the random walk
    #
    # Remember the constraints:
    # - max_total_edges: Maximum number of edges in the graph
    # - max_edges_per_node: Maximum edges per node
    # - All nodes must remain in the graph
    # - Edge weights must be positive and ≤ 10

    # ---------------------------------------------------------------
    # EXAMPLE: Simple strategy to meet edge count constraint
    # This is just a basic example - you should implement a more
    # sophisticated strategy based on query analysis!
    # ---------------------------------------------------------------
    
    # dict mapping query node to frequency of query
    query_frequencies_dict = query_frequencies()

    # dict mapping each node to its shortest distance from a queried node
    shortest_distances_to_targets = {}

    for node, edges in optimized_graph.items():
        shortest_distance = bfs_shortest_path(optimized_graph, node, set(query_frequencies_dict.keys()))
        shortest_distances_to_targets[node] = shortest_distance
        for neighbor, weight in edges.items():
            if neighbor in query_frequencies_dict:
                # scale the edge weight by the frequency of the target node
                frequency_const = query_frequencies_dict[neighbor]/17 # 17 is the max number of queries for a target in this dataset
                new_weight = MAX_WEIGHT/2 * (1 + frequency_const) # all edges leading to targeted nodes have weight [5, 10]
            else:
                # neighbor is never queried
                # should adjust this based on how often this node appears in the path leading to popular nodes
                # for now, just set to random number (0, 5)
                new_weight = random.uniform(0, 5)
            optimized_graph[node][neighbor] = new_weight
    
    # Add edges to nodes that are farthest from queried nodes
    sorted_nodes_by_distance = sorted(shortest_distances_to_targets.items(), key=lambda x: x[1], reverse=True)
    # print('sorted nodes by distance: ', sorted_nodes_by_distance)
    threshold = 5

    # Calculate the probability of selecting each target node based on its query frequency
    total_queries = sum(query_frequencies_dict.values())
    target_node_probabilities = {node: freq/total_queries for node, freq in query_frequencies_dict.items()}

    # Normalize the probabilities so that they sum up to 1
    total_probability = sum(target_node_probabilities.values())
    for node in target_node_probabilities:
        target_node_probabilities[node] /= total_probability

    for node, shortest_distance in sorted_nodes_by_distance:
        if shortest_distance <= threshold:
            break # stop assigning edges
        
        # Choose a target node based on its query frequency (proportional selection)
        target_node = random.choices(list(target_node_probabilities.keys()), 
                                    weights=target_node_probabilities.values(), k=1)[0]
        
        # Add the edge from the node to the selected target node
        if target_node not in optimized_graph[node]:
            frequency_const = query_frequencies_dict[target_node]/17 # 17 is the max number of queries for a target in this dataset
            new_weight = min(MAX_WEIGHT/2 * (1 + frequency_const), 10)
            optimized_graph[node][target_node] = new_weight 
        
    # Count total edges in the initial graph
    total_edges = sum(len(edges) for edges in optimized_graph.values())    

    # If we exceed the limit, we need to prune edges
    if total_edges > max_total_edges:
        optimized_graph = prune_edges(optimized_graph, query_frequencies_dict)


    # =============================================================
    # End of your implementation
    # =============================================================

    # Verify constraints
    if not verify_constraints(optimized_graph, max_edges_per_node, max_total_edges):
        print("WARNING: Your optimized graph does not meet the constraints!")
        print("The evaluation script will reject it. Please fix the issues.")

    return optimized_graph


if __name__ == "__main__":
    # Get file paths
    initial_graph_file = os.path.join(project_dir, "data", "initial_graph.json")
    results_file = os.path.join(project_dir, "data", "initial_results.json")
    output_file = os.path.join(
        project_dir, "candidate_submission", "optimized_graph.json"
    )

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Loading initial graph from {initial_graph_file}")
    initial_graph = load_graph(initial_graph_file)

    print(f"Loading query results from {results_file}")
    results = load_results(results_file)

    print("Optimizing graph...")
    optimized_graph = optimize_graph(initial_graph, results)

    print(f"Saving optimized graph to {output_file}")
    save_graph(optimized_graph, output_file)

    print("Done! Optimized graph has been saved.")
