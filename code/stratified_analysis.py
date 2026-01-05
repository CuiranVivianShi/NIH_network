import leidenalg
import igraph as ig
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors  # Ensure this is included
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load the adjacency matrix data
data1 = pd.read_csv('/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/10year_2006-2015/adjacent_matrix_largest_component_2006-2015.csv')
data2 = pd.read_csv('/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2016-2020/adjacent_matrix_largest_component_2016-2020.csv')
data3 = pd.read_csv('/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2021-2023/adjacent_matrix_largest_component_2021-2023.csv')
A1 = data1.to_numpy()
A2 = data2.to_numpy()
A3 = data3.to_numpy()


A_square1 = A1[:, 1:]  # Exclude the first column
PI_names1 = data1.columns[1:]  # Assuming the first column contains labels which are PI names
A_square2 = A2[:, 1:]
PI_names2 = data2.columns[1:]
A_square3 = A3[:, 1:]
PI_names3 = data3.columns[1:]


# Create an undirected graph from the adjacency matrix
graph1 = ig.Graph.Adjacency((A_square1 > 0).tolist(), mode="undirected")
graph1.es['weight'] = A_square1[A_square1.nonzero()]
graph1.vs['label'] = PI_names1

graph2 = ig.Graph.Adjacency((A_square2 > 0).tolist(), mode="undirected")
graph2.es['weight'] = A_square2[A_square2.nonzero()]
graph2.vs['label'] = PI_names2

graph3 = ig.Graph.Adjacency((A_square3 > 0).tolist(), mode="undirected")
graph3.es['weight'] = A_square3[A_square3.nonzero()]
graph3.vs['label'] = PI_names3


# Leiden algorithm for community detection
resolution_parameter = 0.0001
n_iterations = 20
seed = 42
max_comm_size = 850

partition1 = leidenalg.find_partition(graph1,
                                    leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=0.01,
                                    max_comm_size=max_comm_size,
                                    n_iterations=n_iterations,
                                    seed=seed)
partition2 = leidenalg.find_partition(graph2,
                                    leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=resolution_parameter,
                                    max_comm_size=max_comm_size,
                                    n_iterations=n_iterations,
                                    seed=seed)
partition3 = leidenalg.find_partition(graph3,
                                    leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=resolution_parameter,
                                    max_comm_size=max_comm_size,
                                    n_iterations=n_iterations,
                                    seed=seed)

# Get the clusters
clusters1 = partition1.membership
num_clusters1 = max(partition1.membership) + 1

clusters2 = partition2.membership
num_clusters2 = max(partition2.membership) + 1

clusters3 = partition3.membership
num_clusters3 = max(partition3.membership) + 1




# Iterate over each cluster
def process_clusters(graph, partition):
    all_adj_matrices = {}
    all_collaborations = {}
    top_PIs_by_adjusted_degree = {}

    clusters = partition.membership
    num_clusters = max(clusters) + 1

    for cluster_index in range(num_clusters):
        members = [index for index, cluster_id in enumerate(clusters) if cluster_id == cluster_index]
        labels = [graph.vs['label'][index] for index in members]
        subgraph = graph.subgraph(members)
        adj_matrix = subgraph.get_adjacency(attribute='weight')
        adj_matrix_df = pd.DataFrame(np.array(adj_matrix.data), index=labels, columns=labels)
        all_adj_matrices[cluster_index] = adj_matrix_df  # Store adjacency matrix DataFrame

        # Calculate the adjusted node degrees
        degrees = subgraph.degree()
        degree_df = pd.DataFrame({'PI': labels, 'Degree': degrees})
        degree_df['Adjusted Degree'] = zscore(degree_df['Degree'])  # Standardize degrees
        sorted_degree_df = degree_df.sort_values(by='Adjusted Degree', ascending=False).head(10)
        top_PIs_by_adjusted_degree[cluster_index] = sorted_degree_df  # Store top PIs by adjusted degree

        collaborations = []
        for i in range(len(adj_matrix_df)):
            for j in range(i + 1, len(adj_matrix_df)):
                if adj_matrix_df.iat[i, j] > 0:  # Change from ==1 to >0 to account for all types of weights
                    collaborations.append((adj_matrix_df.index[i], adj_matrix_df.columns[j]))

        collab_df = pd.DataFrame(collaborations, columns=['PI1', 'PI2'])
        all_collaborations[cluster_index] = collab_df  # Store collaborations DataFrame

    return all_adj_matrices, all_collaborations, top_PIs_by_adjusted_degree


# Example usage:
all_adj_matrices1, all_collaborations1, top_PIs_by_adjusted_degree1 = process_clusters(graph1, partition1)
all_adj_matrices2, all_collaborations2, top_PIs_by_adjusted_degree2 = process_clusters(graph2, partition2)
all_adj_matrices3, all_collaborations3, top_PIs_by_adjusted_degree3 = process_clusters(graph3, partition3)



# Match with combined df
def find_cluster_matches(all_collaborations, expanded_combined_df, matched_indices, verbose=False):
    """
    Find and store matches from expanded_combined_df based on collaborations in each cluster.

    Parameters:
        all_collaborations (dict): A dictionary where each key is a cluster index and each value is a dict with 'PI1' and 'PI2'.
        expanded_combined_df (DataFrame): The main dataframe to search for matches.
        matched_indices (set): Set of indices that have already been matched. If None, initializes an empty set.
        verbose (bool): If True, prints matched rows per cluster.

    Returns:
        all_matches (dict): Dictionary of matched DataFrames keyed by cluster index.
        matched_indices (set): Updated set of matched row indices.
    """
    if matched_indices is None:
        matched_indices = set()

    all_matches = {}

    for cluster_index, cluster_collaborations in all_collaborations.items():
        condition = (
            (expanded_combined_df['Contact PI / Project Leader'].isin(cluster_collaborations['PI1']) &
             expanded_combined_df['Other PI or Project Leader(s)'].isin(cluster_collaborations['PI2'])) |
            (expanded_combined_df['Contact PI / Project Leader'].isin(cluster_collaborations['PI2']) &
             expanded_combined_df['Other PI or Project Leader(s)'].isin(cluster_collaborations['PI1']))
        )

        matches = expanded_combined_df[condition & ~expanded_combined_df.index.isin(matched_indices)].drop_duplicates()

        if not matches.empty:
            all_matches[cluster_index] = matches
            matched_indices.update(matches.index)

            if verbose:
                print(f"Matched rows in combined_df for Cluster {cluster_index + 1}:")
                print(matches)
                print("\n")

    return all_matches, matched_indices


all_matches1, matched_indices1 = find_cluster_matches(
    all_collaborations1,
    expanded_combined_df,
    matched_indices=set(),  # Or an existing set if resuming
    verbose=True
)

all_matches2, matched_indices2 = find_cluster_matches(
    all_collaborations2,
    expanded_combined_df,
    matched_indices=set(),
    verbose=True
)
all_matches3, matched_indices3 = find_cluster_matches(
    all_collaborations3,
    expanded_combined_df,
    matched_indices=set(),
    verbose=True
)



# Topic modeling for each cluster at each time point
def run_topic_modeling_per_cluster(all_matches, model, topic_model, preprocess_text_func, verbose=True):
    """
    Apply topic modeling for each cluster's matched texts.

    Parameters:
        all_matches (dict): Cluster index → DataFrame of matched rows
        model: Sentence embedding model with .embed_sentence() method
        topic_model: BERTopic model instance with .transform() method
        preprocess_text_func (function): Function to clean text
        verbose (bool): If True, prints topic counts per cluster

    Returns:
        topic_results (list): List of dictionaries with topic modeling results per cluster
    """
    import numpy as np
    np.random.seed(42)

    topic_results = []

    for i in sorted(all_matches.keys()):
        if verbose:
            print(f"\nProcessing Cluster {i}:")
        cluster_matches = all_matches[i]

        # Combine and clean text
        cluster_matches_text = cluster_matches[['Project Title', 'Project Abstract']].astype(str).drop_duplicates()
        cluster_matches_text['Project Text'] = cluster_matches_text['Project Title'] + " " + cluster_matches_text['Project Abstract']
        cluster_matches_text['cleaned_text'] = cluster_matches_text['Project Text'].apply(preprocess_text_func)

        # Embed
        cluster_matches_text['embeddings'] = cluster_matches_text['cleaned_text'].apply(model.embed_sentence)
        cluster_matches_text['embeddings'] = cluster_matches_text['embeddings'].apply(
            lambda x: x[0] if len(x) > 0 else None)

        # Filter valid embeddings
        valid_embeddings = [e for e in cluster_matches_text['embeddings'] if e is not None]
        if len(valid_embeddings) == 0:
            if verbose:
                print(f"Skipping Cluster {i} (no valid embeddings).")
            continue

        cluster_embeddings_array = np.vstack(valid_embeddings)

        # Topic modeling
        topics, probabilities = topic_model.transform(cluster_matches_text['cleaned_text'], cluster_embeddings_array)

        cluster_matches_text['predicted_topic'] = topics
        topic_counts = cluster_matches_text['predicted_topic'].value_counts()

        if verbose:
            print(topic_counts)

        topic_results.append({
            'cluster': i,
            'topics': topics,
            'probabilities': probabilities,
            'topic_counts': topic_counts
        })

    return topic_results

# For your first set of matches
topic_results1 = run_topic_modeling_per_cluster(
    all_matches=all_matches1,
    model=model,
    topic_model=topic_model,
    preprocess_text_func=preprocess_text,
    verbose=True
)

# Then for second set:
topic_results2 = run_topic_modeling_per_cluster(
    all_matches=all_matches2,
    model=model,
    topic_model=topic_model,
    preprocess_text_func=preprocess_text,
    verbose=True
)

topic_results3 = run_topic_modeling_per_cluster(
    all_matches=all_matches3,
    model=model,
    topic_model=topic_model,
    preprocess_text_func=preprocess_text,
    verbose=True
)


import pandas as pd

def get_topic_percentages(topic_results, label):
    """
    Compute percentage of each topic within each cluster from topic modeling results.

    Parameters:
        topic_results (list): Output from run_topic_modeling_per_cluster
        label (str): A label for identifying the result set (e.g. "Set1")

    Returns:
        DataFrame: A long-format table with cluster, topic, count, percentage, and source label
    """
    records = []

    for result in topic_results:
        cluster_id = result['cluster']
        topic_counts = result['topic_counts'].drop(-1, errors='ignore')  # optionally drop outliers
        total = topic_counts.sum()

        for topic, count in topic_counts.items():
            percentage = (count / total) * 100
            records.append({
                'Set': label,
                'Cluster': cluster_id,
                'Topic': topic,
                'Count': count,
                'Percentage': round(percentage, 2)
            })

    return pd.DataFrame(records)


# Collect all percentages from the 3 sets
df1 = get_topic_percentages(topic_results1, label="Set1")
df2 = get_topic_percentages(topic_results2, label="Set2")
df3 = get_topic_percentages(topic_results3, label="Set3")

# Combine into one DataFrame
all_topic_percentages = pd.concat([df1, df2, df3], ignore_index=True)

all_topic_percentages.to_csv('all_topic_percentages.csv', index=False)

# Display or save
print(all_topic_percentages.head())
# all_topic_percentages.to_csv("topic_percentages_by_cluster.csv", index=False)

print(df1[df1['Cluster'] == 1]);
print(df2[df2['Cluster'] == 1]);
print(df3[df3['Cluster'] == 1]);




"""
Step 1
"""

# Get PI names for each group
pis_g1 = PI_names1
pis_g2 = PI_names2
pis_g3 = PI_names3

# Identify intersections and union relevant for Sankey diagram
intersection_g1_g2 = pis_g1.intersection(pis_g2)
intersection_g2_g3 = pis_g2.intersection(pis_g3)
relevant_pis = intersection_g1_g2.union(intersection_g2_g3)

# Generate cluster labels for each time interval
label1 = [f"A{i+1}" for i in range(num_clusters1)]
label2 = [f"B{i+1}" for i in range(num_clusters2)]
label3 = [f"C{i+1}" for i in range(num_clusters3)]
# Combine all labels into one list
label = label1 + label2 + label3


# Define a function to extract PI names from the cluster adjacency matrices
def extract_pi_names_from_clusters(adj_matrices):
    cluster_pi_names = {}
    for cluster_id, matrix in adj_matrices.items():
        # The index (or columns, since it's symmetrical) of the matrix contains the PI names
        pi_names = matrix.index.tolist()  # or matrix.columns.tolist()
        cluster_pi_names[cluster_id] = pi_names
    return cluster_pi_names

# Use the function on the adjacency matrices of the first time interval
pi_names_by_cluster1 = extract_pi_names_from_clusters(all_adj_matrices1)
pi_names_by_cluster2 = extract_pi_names_from_clusters(all_adj_matrices2)
pi_names_by_cluster3 = extract_pi_names_from_clusters(all_adj_matrices3)





"""
Step 2
"""
def filter_pi_names_by_relevance(pi_names_by_cluster, relevant_pis_set):
    filtered_pi_names = {}
    for cluster_id, pi_names in pi_names_by_cluster.items():
        # Using set intersection to filter names
        filtered_names = list(set(pi_names).intersection(relevant_pis_set))
        if filtered_names:  # Only add the cluster if there are any relevant PIs
            filtered_pi_names[cluster_id] = filtered_names
    return filtered_pi_names

# Convert relevant_pis Index to a set for efficient intersection
relevant_pis_set = set(relevant_pis)

# Filter the PI names in the first time interval
filtered_pi_names_by_cluster1 = filter_pi_names_by_relevance(pi_names_by_cluster1, relevant_pis_set)
filtered_pi_names_by_cluster2 = filter_pi_names_by_relevance(pi_names_by_cluster2, relevant_pis_set)
filtered_pi_names_by_cluster3 = filter_pi_names_by_relevance(pi_names_by_cluster3, relevant_pis_set)


def find_overlapping_pis(cluster_dict1, cluster_dict2):
    """ Finds and returns overlapping PIs between all combinations of clusters from two different cluster dictionaries. """
    overlaps = {}
    for key1, pis1 in cluster_dict1.items():
        for key2, pis2 in cluster_dict2.items():
            intersection = list(set(pis1) & set(pis2))
            if intersection:
                overlaps[(key1, key2)] = intersection
    return overlaps

# Assuming filtered_pi_names_by_cluster1 and filtered_pi_names_by_cluster2 have been defined earlier
overlaps_pis_G1_G2 = find_overlapping_pis(filtered_pi_names_by_cluster1, filtered_pi_names_by_cluster2)
overlaps_pis_G2_G3 = find_overlapping_pis(filtered_pi_names_by_cluster2, filtered_pi_names_by_cluster3)

# Output the results
print("Overlaps between all combinations of clusters from Interval 1 and Interval 2:")
for (i, j), pis in overlaps_pis_G1_G2.items():
    print(f"Overlap between Cluster {i} in Interval 1 and Cluster {j} in Interval 2: {pis}")

print("Overlaps between all combinations of clusters from Interval 2 and Interval 3:")
for (i, j), pis in overlaps_pis_G2_G3.items():
    print(f"Overlap between Cluster {i} in Interval 2 and Cluster {j} in Interval 3: {pis}")

# Check for empty combinations
list(set(filtered_pi_names_by_cluster1[0]) & set(filtered_pi_names_by_cluster2[0]))



def count_pis_in_overlaps(overlaps):
    """ Returns the size of overlapping PIs for each cluster combination. """
    sizes = {}
    for key, pis in overlaps.items():
        sizes[key] = len(pis)  # Count the number of PIs in each overlap
    return sizes

# Assuming overlaps_pis_G1_G2 and overlaps_pis_G2_G3 have been defined as per your previous function
sizes_overlaps_pis_G1_G2 = count_pis_in_overlaps(overlaps_pis_G1_G2)
sizes_overlaps_pis_G2_G3 = count_pis_in_overlaps(overlaps_pis_G2_G3)

# Print the sizes for visualization or further analysis
print("Sizes of overlaps between clusters from G1 to G2:")
for key, size in sizes_overlaps_pis_G1_G2.items():
    print(f"Overlap between Cluster {key[0]} in G1 and Cluster {key[1]} in G2 has {size} PIs")

print("\nSizes of overlaps between clusters from G2 to G3:")
for key, size in sizes_overlaps_pis_G2_G3.items():
    print(f"Overlap between Cluster {key[0]} in G2 and Cluster {key[1]} in G3 has {size} PIs")


"""
Step 3. Sankey plot
"""
def map_and_count_combinations(sizes_overlaps, labels_source, labels_target):
    combinations = []
    for (source_index, target_index), size in sizes_overlaps.items():
        source_label = labels_source[source_index]
        target_label = labels_target[target_index]
        combinations.append((source_label, target_label, size))
    return combinations

# Mapping and counting combinations for G1 to G2 and G2 to G3
combinations_G1_G2 = map_and_count_combinations(sizes_overlaps_pis_G1_G2, label1, label2)
combinations_G2_G3 = map_and_count_combinations(sizes_overlaps_pis_G2_G3, label2, label3)

# Combine all combinations
all_combinations = combinations_G1_G2 + combinations_G2_G3

# Output the total number of combinations and the details for each
print(f"Total combinations: {len(all_combinations)}")
for source, target, size in all_combinations:
    print(f"Source: {source}, Target: {target}, Number of PIs: {size}")





import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Extract all unique labels and assign indices
unique_labels = sorted(set([item for sublist in all_combinations for item in sublist[:2]]))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

# Create the link dictionary structure
link = {
    'source': [],
    'target': [],
    'value': [],
    'color': []  # Adding color to the dictionary structure
}

# Define a color map and generate distinct colors based only on unique sources
#cmap = plt.get_cmap('tab20')  # 'tab20' has 20 distinct colors, use other cmaps if more colors are needed
# Ensure we have enough colors for each unique source
#source_colors = {label_to_index[label]: cmap(i % 20) for i, label in enumerate(unique_labels)}
# Convert RGBA tuples to HEX
#colors_hex = {index: mcolors.rgb2hex(color) for index, color in source_colors.items()}

cmap = plt.get_cmap("YlGnBu")
norm = mcolors.Normalize(vmin=0, vmax=len(unique_labels) - 1)
source_colors = {label_to_index[label]: cmap(norm(i)) for i, label in enumerate(unique_labels)}
colors_hex = {index: mcolors.rgb2hex(color) for index, color in source_colors.items()}

# Populate the link dictionary
for source_label, target_label, value in all_combinations:
    source_index = label_to_index[source_label]
    target_index = label_to_index[target_label]
    link['source'].append(source_index)
    link['target'].append(target_index)
    link['value'].append(value)
    link['color'].append(colors_hex[source_index])  # Assign the color based on source index

# Output the completed link dictionary to verify
print("Link Dictionary:")
print(f"Sources: {link['source']}")
print(f"Targets: {link['target']}")
print(f"Values: {link['value']}")
print(f"Colors: {link['color']}")

#colors_label = [cmap(i) for i in range(len(label))]  # Generate a list of RGBA color tuples
#colors_hex_label = [mcolors.rgb2hex(color) for color in colors_label]

colors_hex_label = [colors_hex[i] for i in range(len(label))]


label = [
    "Genomics & Neurobiology",
    "Genetics & Pain",
    "Public Health & Genomics",
    "Genomics & Clinical Links",
    "Epidemiology & Infection",
    "Neuroscience & Systems Biology",
    "Brain & Diagnostics",
    "Genomics & Health Research",
    "Epidemiology & Disease Biology",
    "Pop Health & Diagnostics",
    "Genomics & Multisystem",
    "Genomics & Infection",
    "Behavioral & Public Health"
]

import plotly.graph_objects as go

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=label,
        color=colors_hex_label
    ),
    link=link
)])

fig.update_layout(
    font=dict(size=22),
    annotations=[
        dict(
            x=0.05,
            y=1.05,
            text="<b>Period 1: 2006–2015</b>",
            showarrow=False,
            font=dict(size=25, color="black")
        ),
        dict(
            x=0.5,
            y=1.05,
            text="<b>Period 2: 2016–2020</b>",
            showarrow=False,
            font=dict(size=25, color="black")
        ),
        dict(
            x=0.95,
            y=1.05,
            text="<b>Period 3: 2021–2023</b>",
            showarrow=False,
            font=dict(size=25, color="black")
        )
    ]
)

fig.write_image("sankey_diagram.png", width=1600, height=1200, scale=2)