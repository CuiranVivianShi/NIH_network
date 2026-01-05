import pandas as pd
import networkx as nx

# Load the datasets
adj_matrix = pd.read_csv('adjacency_matrix_largest_component.csv', index_col=0)
pi_names = pd.read_csv('PI_name_largest_component.csv')
edges = pd.read_csv('edges_largest_component.csv')


# Create a graph from the adjacency matrix
G = nx.from_pandas_adjacency(adj_matrix)

# Alternatively, if using edges and nodes explicitly
# G = nx.from_pandas_edgelist(edges, source='source', target='target')
# Add nodes explicitly if they have attributes or isolated nodes
# G.add_nodes_from(pi_names['PI Names'])

# This is a placeholder for the actual analysis function
def perform_mixed_membership(graph):
    # Implement or call the mixed-membership estimation model
    # For demonstration, returning empty results
    return {}

# Perform the estimation
results = perform_mixed_membership(G)


import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Serialize the analysis results
json_str = json.dumps(results, cls=NumpyEncoder)

# Write to a file
with open('mixed_membership_results.json', 'w') as f:
    f.write(json_str)

