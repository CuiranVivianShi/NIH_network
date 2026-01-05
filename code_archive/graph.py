from read_data import get_combined_df
from read_data import get_unique_pis
from itertools import combinations
import pandas as pd
from collections import namedtuple
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from jaal import Jaal
import dash_html_components as html
from dash import html
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt# Example adjacency matrix
import csv


#combined_df = get_combined_df()
#print(combined_df.head())


# Build graph
#Graph = namedtuple("Graph", ["nodes", "edges"])
#all_unique_pis = get_unique_pis()
nodes = list(all_unique_pis)


# A function to clean and split PI names
def process_pi_names(pi_str):
    if pd.isna(pi_str):
        return []
    return [name.strip() for name in pi_str.split(';') if name.strip()]

import ast




edges = []

# Iterate over each project (row in DataFrame)
# Later I found out that one project number could have multiple rows because it was funded by multiple sources,
# so there are duplication in edges but does not cause problems in our analysis because we use unweighted graph
for index, row in combined_df.iterrows():
    # Combine all PIs from both columns into one list and remove any potential empty strings
    pis = [row['Contact PI / Project Leader'].strip()] if pd.notna(row['Contact PI / Project Leader']) else []
    other_pis = process_pi_names(row['Other PI or Project Leader(s)'])
    pis.extend(other_pis)

    # Generate all combinations of PIs within this project as edges
    project_edges = list(combinations(pis, 2))

    # Add these edges to the main list
    edges.extend(project_edges)

# Print or process the edges as needed
print("Total edges:", len(edges)) # Total edges: 154175
for edge in edges[:10]:  # Print only the first 10
    print(edge)


# Adjacent matrix


# Create an empty DataFrame with nodes as index and columns
adj_matrix = pd.DataFrame(0, index=nodes, columns=nodes)

# Populate the adjacency matrix
for start, end in edges:
    adj_matrix.loc[start, end] += 1
    adj_matrix.loc[end, start] += 1  # Uncomment this line if the graph is undirected

# Print the adjacency matrix
print(adj_matrix.head())
#print(adj_matrix.loc['FEDER, ADRIANA', 'PIETRZAK, ROBERT H'])



#G_nx = nx.DiGraph(adj_matrix.values)
#nx.draw(G_nx)


# Draw graph
#net = Network(notebook=True)
#net.from_nx(G_nx)
#net.show("example.html")


nodes_df = pd.DataFrame(list(nodes), columns=['PI Names'])
edges_df = pd.DataFrame(list(edges), columns=['source', 'target'])


#Jaal(edges_df[0:10], nodes_df[0:10]).plot()

edges_df.to_csv('edges_dec31.csv', index=False)
nodes_df.to_csv('nodes_dec31.csv', index=False)
adj_matrix.to_csv('adjacent_dec31.csv', index=True)





import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


A = adj_matrix.to_numpy()

# Create a graph from the adjacency matrix
G = nx.from_numpy_array(A)

# Find all connected components
connected_components = list(nx.connected_components(G))

# Calculate the size of each connected component
component_sizes = [len(component) for component in connected_components]
sorted(component_sizes, reverse=True)
max(component_sizes)

# Plot the histogram of the sizes
#plt.hist(component_sizes, bins=range(1, max(component_sizes) + 2), align='left', edgecolor='black')
#plt.xlabel('Component Size')
#plt.ylabel('Frequency')
#plt.title('Histogram of Connected Component Sizes')
#plt.show()



#with open('size', 'w') as f:
#    write = csv.writer(f)
#    write.writerow(component_sizes)


# Find the largest connected component
largest_cc = max(nx.connected_components(G), key=len)
subgraph = G.subgraph(largest_cc)

# Draw the largest component
#pos = nx.spring_layout(subgraph)
#nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=15)

# Return the edges of the largest component
edges_largest_component = list(subgraph.edges())

#plt.show(), edges_largest_component

# Assuming 'nodes' is your list of PI names and 'edges_largest_component' contains the edge indices
edges_largest_component_named = [(nodes[start], nodes[end]) for start, end in edges_largest_component]

# Print the first few to check
#print(edges_largest_component_named[:10])
#print(edges_largest_component[:10])


edges_largest_component_named_df = pd.DataFrame(edges_largest_component_named, columns=['source', 'target'])

#Jaal(edges_df[0:10], nodes_df[0:10]).plot()

#edges_df.to_csv('edges.csv', index=False)
edges_largest_component_named_df.to_csv('edges_largest_component_dec31.csv', index=False)

# Extract unique PIs using in the largest component
unique_pis_largest = set()
for edge in edges_largest_component_named:
    unique_pis_largest.update(edge)


pis_largest_df = pd.DataFrame(list(unique_pis_largest), columns=['PI Names'])
pis_largest_df.to_csv('nodes_largest_component_dec31.csv', index=False)



# Adjacent matrix for the largest component
# Create an empty DataFrame with nodes as index and columns
adj_matrix_largest = pd.DataFrame(0, index=list(unique_pis_largest), columns=list(unique_pis_largest))

# Populate the adjacency matrix
for start, end in edges_largest_component_named:
    adj_matrix_largest.loc[start, end] += 1
    adj_matrix_largest.loc[end, start] += 1  # Uncomment this line if the graph is undirected

# Print the adjacency matrix
adj_matrix_largest.to_csv('adjacent_matrix_largest_component_dec31.csv', index=True)