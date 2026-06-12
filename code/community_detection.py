import leidenalg
import igraph as ig
import pandas as pd
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from collections import defaultdict


"""
Section 1
find clusters based on adjacency matrix
"""

# Load the adjacency matrix data
data = pd.read_csv('adjacency_matrix_largest_component.csv')

A = data.to_numpy()
A_square = A[:, 1:]  # Exclude the first column
PI_names = data.columns[1:]  # The first column contains labels which are PI names

# Create an undirected graph from the adjacency matrix
graph = ig.Graph.Adjacency((A_square > 0).tolist(), mode="undirected")
graph.es['weight'] = A_square[A_square.nonzero()]
graph.vs['label'] = PI_names

# Leiden algorithm for community detection
resolution_parameter = 0.0001
n_iterations = 20
seed = 42
max_comm_size = 850

partition = leidenalg.find_partition(graph,
                                    leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=resolution_parameter,
                                    max_comm_size=max_comm_size,
                                    n_iterations=n_iterations,
                                    seed=seed)

# ======================================
# Sensitivity Analysis with Leiden Algorithm
# ======================================
# Try different resolutions
resolutions = [0.00001, 0.0001, 0.001, 0.01]

leiden_partitions = {}
for res in resolutions:
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=res,
        max_comm_size=850,
        n_iterations=20,
        seed=42
    )
    leiden_partitions[res] = partition


def summarize_partition(partition, resolution=None):
    membership = partition.membership  # cluster assignment list
    n_clusters = len(partition)         # number of clusters
    sizes = partition.sizes()            # size of each cluster
    modularity = partition.quality()     # modularity score

    if resolution is not None:
        print(f"--- Resolution {resolution} ---")
    print(f"Number of clusters: {n_clusters}")
    print(f"Cluster sizes: {sizes}")
    print(f"Modularity: {modularity:.4f}")
    print("\n")  # extra line for readability

    return {
        'resolution': resolution,
        'n_clusters': n_clusters,
        'sizes': sizes,
        'modularity': modularity,
        'membership': membership
    }


# Summarize all partitions
partition_summaries = {}

for res, partition in leiden_partitions.items():
    summary = summarize_partition(partition, resolution=res)
    partition_summaries[res] = summary



# Louvain algorithm
partition_louvain = leidenalg.find_partition(graph,
                                    leidenalg.RBConfigurationVertexPartition,
                                    seed=seed)


membership = partition_louvain.membership  # cluster assignment
n_clusters = len(partition_louvain)
modularity = partition_louvain.modularity

print(f"Number of clusters: {n_clusters}")
print(f"Modularity: {modularity:.4f}")



# Get the clusters
clusters = partition.membership
num_clusters = max(partition.membership) + 1

# Prepare dictionaries to store DataFrames for each cluster
all_adj_matrices = {}
all_collaborations = {}

# Iterate over each cluster
for cluster_index in range(num_clusters):
    members = [index for index, cluster_id in enumerate(clusters) if cluster_id == cluster_index]
    labels = [graph.vs['label'][index] for index in members]
    subgraph = graph.subgraph(members)
    adj_matrix = subgraph.get_adjacency(attribute='weight')
    adj_matrix_df = pd.DataFrame(np.array(adj_matrix.data), index=labels, columns=labels)
    all_adj_matrices[cluster_index] = adj_matrix_df  # Store adjacency matrix DataFrame

    collaborations = []
    for i in range(len(adj_matrix_df)):
        for j in range(i + 1, len(adj_matrix_df)):
            if adj_matrix_df.iat[i, j] == 1:
                collaborations.append((adj_matrix_df.index[i], adj_matrix_df.columns[j]))

    collab_df = pd.DataFrame(collaborations, columns=['PI1', 'PI2'])
    all_collaborations[cluster_index] = collab_df  # Store collaborations DataFrame

# Example of accessing stored DataFrames
for cluster_index in all_adj_matrices:
    print(f"Adjacency Matrix for Cluster {cluster_index + 1}:")
    print(all_adj_matrices[cluster_index])
    print(f"Collaborating PIs in Cluster {cluster_index + 1}:")
    print(all_collaborations[cluster_index])
    print("\n")


# Access the adjacency matrix for Cluster 1
cluster_1_adj_matrix = all_adj_matrices[0]
print("Adjacency Matrix for Cluster 1:")
print(cluster_1_adj_matrix)

# Access the collaborations list for Cluster 1
cluster_1_collaborations = all_collaborations[0]
print("Collaborating PIs in Cluster 1:")
print(cluster_1_collaborations)


# Save the adjacency matrice
for i in range(19):  # Loop through each cluster index from 0 to 18
    # Access the adjacency matrix for current cluster
    cluster_adj_matrix = all_adj_matrices[i]

    # Print the adjacency matrix (optional, can be commented out if not needed)
    print(f"Adjacency Matrix for Cluster {i + 1}:")
    print(cluster_adj_matrix)

    # Convert the matrix to a DataFrame
    df = pd.DataFrame(cluster_adj_matrix)

    # Save the DataFrame to a CSV file
    df.to_csv(f'cluster_{i + 1}_adj_matrix.csv', index=False)


# Adjusting cluster indices to start from 1 instead of 0
cluster_labels = [x + 1 for x in partition.membership]

# Extract node labels (PI names)
node_labels = graph.vs['label']

# Create a DataFrame with node labels and their corresponding cluster labels
cluster_df = pd.DataFrame({
    'Node_Label': node_labels,
    'Cluster_Label': cluster_labels
})

# Save the DataFrame to a CSV file
cluster_df.to_csv('node_cluster_labels.csv', index=False)

# Generating unique colors using matplotlib's colormap for up to 20 distinct clusters
colors = plt.get_cmap('tab20', max(cluster_labels))  # Ensure there are enough colors

# Map each cluster label to a unique color
color_map = {label: matplotlib.colors.rgb2hex(colors(label-1)) for label in set(cluster_labels)}

# Apply the color map to the DataFrame based on cluster labels
cluster_df['node_color'] = cluster_df['Cluster_Label'].map(color_map)

# The 'type' column is essentially the node labels again
cluster_df['type'] = cluster_df['Cluster_Label']

cluster_df.drop(columns=['Cluster_Label'], inplace=True)
cluster_df.rename(columns={'Node_Label': 'id'}, inplace=True)

# Save the DataFrame to a CSV file
cluster_df.to_csv('final_node_data.csv', index=False)




"""
Section 2
Merge the clusters with the complete dataset to find the grants in each cluster
"""

# Load the data
combined_df = pd.read_csv('combined_df.csv', low_memory=False)

# Normalize the data by splitting and exploding
combined_df['Other PI or Project Leader(s)'] = combined_df['Other PI or Project Leader(s)'].str.split(';')
expanded_combined_df = combined_df.explode('Other PI or Project Leader(s)')

# Clean up whitespace
expanded_combined_df['Contact PI / Project Leader'] = expanded_combined_df['Contact PI / Project Leader'].str.strip()
expanded_combined_df['Other PI or Project Leader(s)'] = expanded_combined_df['Other PI or Project Leader(s)'].str.strip()

# Initialize the dictionary to store matches and a set to track matched row indices
all_matches = {}
matched_indices = set()

# Loop through each cluster's collaborations
for cluster_index, cluster_collaborations in all_collaborations.items():
    condition = (
        (expanded_combined_df['Contact PI / Project Leader'].isin(cluster_collaborations['PI1']) &
         expanded_combined_df['Other PI or Project Leader(s)'].isin(cluster_collaborations['PI2'])) |
        (expanded_combined_df['Contact PI / Project Leader'].isin(cluster_collaborations['PI2']) &
         expanded_combined_df['Other PI or Project Leader(s)'].isin(cluster_collaborations['PI1']))
    )

    # Filter for rows that haven't been matched yet
    matches = expanded_combined_df[condition & ~expanded_combined_df.index.isin(matched_indices)].drop_duplicates()

    # Store the unique matches
    if not matches.empty:
        all_matches[cluster_index] = matches
        # Update the set of matched indices
        matched_indices.update(matches.index)

        # Optionally print or perform further operations with matches
        print(f"Matched rows in combined_df for Cluster {cluster_index + 1}:")
        print(matches)
        print("\n")


# Extract the matched rows from combined_df for Cluster 1
cluster_1_matches = all_matches[0]
print("Matched rows in combined_df for Cluster 1:")
print(cluster_1_matches)



"""
Section 3
Summary Statistics for each cluster
"""

# Dictionaries to store results
top_10_nodes_per_cluster = {}
cluster_nodes = {}
cluster_edges = {}

# Tables to save
cluster_summary = []
top_10_rows = []

for cluster_index in range(num_clusters):

    # -----------------------------
    # Retrieve subgraph for cluster
    # -----------------------------
    members = [
        index
        for index, cluster_id in enumerate(clusters)
        if cluster_id == cluster_index
    ]

    subgraph = graph.subgraph(members)

    # -----------------------------
    # Nodes and edges
    # -----------------------------
    nodes_labels = subgraph.vs["label"]
    n_nodes = subgraph.vcount()
    n_edges = subgraph.ecount()

    cluster_nodes[cluster_index] = list(nodes_labels)

    # Edge list for this cluster
    edge_list = []
    for edge in subgraph.es:
        source = nodes_labels[edge.source]
        target = nodes_labels[edge.target]
        edge_list.append((source, target))

    cluster_edges[cluster_index] = edge_list

    # -----------------------------
    # Degrees
    # -----------------------------
    degrees = subgraph.degree()
    degrees_array = np.array(degrees)

    # Average node degree
    avg_degree = np.mean(degrees_array) if n_nodes > 0 else 0

    # z-score normalized degree: (degree - mean) / std
    mean_deg = np.mean(degrees_array) if n_nodes > 0 else 0
    std_deg = np.std(degrees_array) if n_nodes > 0 else 0

    if std_deg == 0:
        normalized_degrees = np.zeros_like(degrees_array, dtype=float)
    else:
        normalized_degrees = (degrees_array - mean_deg) / std_deg

    # -----------------------------
    # Save cluster summary
    # -----------------------------
    cluster_summary.append({
        "Cluster": cluster_index + 1,
        "Number of Nodes": n_nodes,
        "Number of Edges": n_edges,
        "Average Node Degree": round(avg_degree, 2)
    })

    # -----------------------------
    # Top 10 representative nodes
    # -----------------------------
    node_degree_pairs = list(
        zip(nodes_labels, degrees_array, normalized_degrees)
    )

    node_degree_pairs_sorted = sorted(
        node_degree_pairs,
        key=lambda x: x[2],
        reverse=True
    )

    top_10_nodes = node_degree_pairs_sorted[:10]
    top_10_nodes_per_cluster[cluster_index] = top_10_nodes

    for node, raw_deg, norm_deg in top_10_nodes:
        top_10_rows.append({
            "Cluster": cluster_index + 1,
            "PI": node,
            "Raw Degree": int(raw_deg),
            "Normalized Degree": round(norm_deg, 2)
        })

    # Print results
    print(f"Cluster {cluster_index + 1}:")
    print(f"  Number of nodes: {n_nodes}")
    print(f"  Number of edges: {n_edges}")
    print(f"  Average node degree: {avg_degree:.2f}")
    print("  Top 10 representative PIs:")
    for node, raw_deg, norm_deg in top_10_nodes:
        print(
            f"    {node} - Raw degree: {raw_deg}, "
            f"Normalized degree: {norm_deg:.2f}"
        )
    print("\n")


# -----------------------------
# Convert to DataFrames
# -----------------------------
cluster_summary_df = pd.DataFrame(cluster_summary)
top_10_nodes_df = pd.DataFrame(top_10_rows)

print("Cluster summary:")
print(cluster_summary_df)

print("Top 10 nodes per cluster:")
print(top_10_nodes_df)


# -----------------------------
# Save summary tables
# -----------------------------
cluster_summary_df.to_csv(
    "cluster_summary_zscore_normalized_degree.csv",
    index=False
)

top_10_nodes_df.to_csv(
    "cluster_top10_representative_pis_zscore_normalized_degree.csv",
    index=False
)

