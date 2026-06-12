import re
import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

from read_data import get_combined_df, get_unique_pis


# -----------------------------
# Load data
# -----------------------------
combined_df = get_combined_df()
nodes = sorted(list(get_unique_pis()))


# -----------------------------
# Functions
# -----------------------------
def process_pi_names(pi_str):
    if pd.isna(pi_str):
        return []
    return [name.strip() for name in str(pi_str).split(";") if name.strip()]


def extract_core_project_number(project_number):
    if pd.isna(project_number):
        return None

    project_number = str(project_number).strip()

    # Remove support year suffix, e.g., -01, -02, -05S1
    project_number = re.sub(r"-\d{2}.*$", "", project_number)

    # Remove leading application type digit, e.g., 1R01 -> R01
    project_number = re.sub(r"^\d", "", project_number)

    return project_number


combined_df["Core Project Number"] = combined_df["Project Number"].apply(
    extract_core_project_number
)


# -----------------------------
# Build project-level PI-pair edges
# -----------------------------
edge_records = []

for _, row in combined_df.iterrows():
    core_project = row["Core Project Number"]

    pis = []

    if pd.notna(row["Contact PI / Project Leader"]):
        pis.extend(process_pi_names(row["Contact PI / Project Leader"]))

    pis.extend(process_pi_names(row["Other PI or Project Leader(s)"]))

    # Remove duplicate names within the same grant record
    pis = sorted(set(pis))

    for pi1, pi2 in combinations(pis, 2):
        # Sort pair to make edge undirected
        source, target = sorted([pi1, pi2])
        edge_records.append((core_project, source, target))


edges_project_df = pd.DataFrame(
    edge_records,
    columns=["core_project", "source", "target"]
)

# Remove repeated grant-year records for the same core project and PI pair
edges_project_df = edges_project_df.drop_duplicates(
    subset=["core_project", "source", "target"]
)


# -----------------------------
# Full network edge tables
# -----------------------------

# Unweighted edges: one row per PI pair
edges_unweighted = edges_project_df[["source", "target"]].drop_duplicates()

# Weighted edges: weight = number of unique core projects shared by PI pair
edges_weighted = (
    edges_project_df
    .groupby(["source", "target"])
    .size()
    .reset_index(name="weight")
)


# -----------------------------
# Full adjacency matrices
# -----------------------------

# Unweighted adjacency matrix
adj_matrix = pd.DataFrame(0, index=nodes, columns=nodes, dtype=int)

for _, row in edges_unweighted.iterrows():
    start = row["source"]
    end = row["target"]
    adj_matrix.loc[start, end] = 1
    adj_matrix.loc[end, start] = 1


# Weighted adjacency matrix
adj_matrix_weighted = pd.DataFrame(0, index=nodes, columns=nodes, dtype=int)

for _, row in edges_weighted.iterrows():
    start = row["source"]
    end = row["target"]
    weight = row["weight"]

    adj_matrix_weighted.loc[start, end] = weight
    adj_matrix_weighted.loc[end, start] = weight


# -----------------------------
# Save full network outputs
# -----------------------------
nodes_df = pd.DataFrame(nodes, columns=["PI Names"])

nodes_df.to_csv("nodes.csv", index=False)
edges_unweighted.to_csv("edges_unweighted.csv", index=False)
edges_weighted.to_csv("edges_weighted.csv", index=False)

adj_matrix.to_csv("adjacency_matrix_unweighted.csv", index=True)
adj_matrix_weighted.to_csv("adjacency_matrix_weighted.csv", index=True)


# -----------------------------
# Check weighted edge distribution
# -----------------------------
weights = adj_matrix_weighted.to_numpy().flatten()
weights = weights[weights > 0]

print("Weighted edge distribution:")
print(pd.Series(weights).value_counts().sort_index())


# -----------------------------
# Largest connected component
# Use unweighted graph structure
# -----------------------------
G = nx.from_pandas_adjacency(adj_matrix)

connected_components = list(nx.connected_components(G))
component_sizes = [len(c) for c in connected_components]

print("Number of components:", len(component_sizes))
print("Largest component size:", max(component_sizes))

print("\nComponent size summary:")
print(f"Minimum size: {np.min(component_sizes)}")
print(f"Maximum size: {np.max(component_sizes)}")
print(f"Mean size: {np.mean(component_sizes):.2f}")
print(f"Median size: {np.median(component_sizes):.2f}")

# ---------------------------------
# Distribution table
# ---------------------------------
component_dist = (
    pd.Series(component_sizes)
    .value_counts()
    .sort_index()
)

component_dist_pct = (
    100 * component_dist / component_dist.sum()
)

print("\nComponent size distribution (counts):")
print(component_dist)

print("\nComponent size distribution (%):")
print(component_dist_pct.round(2))


largest_cc = max(connected_components, key=len)
nodes_lcc = sorted(list(largest_cc))



# -----------------------------
# LCC edge tables
# -----------------------------
edges_unweighted_lcc = edges_unweighted[
    edges_unweighted["source"].isin(nodes_lcc) &
    edges_unweighted["target"].isin(nodes_lcc)
].copy()

edges_weighted_lcc = edges_weighted[
    edges_weighted["source"].isin(nodes_lcc) &
    edges_weighted["target"].isin(nodes_lcc)
].copy()


# -----------------------------
# LCC adjacency matrices
# -----------------------------
adj_matrix_lcc = adj_matrix.loc[nodes_lcc, nodes_lcc]
adj_matrix_weighted_lcc = adj_matrix_weighted.loc[nodes_lcc, nodes_lcc]


# -----------------------------
# Save LCC outputs
# -----------------------------
nodes_lcc_df = pd.DataFrame(nodes_lcc, columns=["PI Names"])

nodes_lcc_df.to_csv("nodes_largest_component.csv", index=False)
edges_unweighted_lcc.to_csv("edges_unweighted_largest_component.csv", index=False)
edges_weighted_lcc.to_csv("edges_weighted_largest_component.csv", index=False)

adj_matrix_lcc.to_csv("adjacency_matrix_unweighted_largest_component.csv", index=True)
adj_matrix_weighted_lcc.to_csv("adjacency_matrix_weighted_largest_component.csv", index=True)


# =====================================================
# Additional outputs for Reviewer Q4:
# 1) Histogram of PI--PI edge weights
# 2) Hypergraph / grant--PI incidence table and histogram of hyperedge sizes
# =====================================================

# -----------------------------
# 1. PI--PI edge weight distribution
# -----------------------------
edge_weight_values = edges_weighted["weight"]

edge_weight_counts = (
    edge_weight_values
    .value_counts()
    .sort_index()
)

edge_weight_percent = (
    100 * edge_weight_counts / edge_weight_counts.sum()
)

print("\nPI--PI edge weight distribution (counts):")
print(edge_weight_counts)

print("\nPI--PI edge weight distribution (%):")
print(edge_weight_percent.round(2))

plt.figure(figsize=(8, 5))

plt.hist(
    edge_weight_values,
    bins=np.arange(0.5, edge_weight_values.max() + 1.5, 1),
    edgecolor="black"
)

plt.xticks(range(1, edge_weight_values.max() + 1))

plt.xlabel("Edge weight: number of unique core projects shared by a PI pair")
plt.ylabel("Number of PI pairs")
plt.title("Distribution of PI--PI Edge Weights")

plt.tight_layout()
plt.savefig("hist_edge_weights.png", dpi=300)
plt.show()


# -----------------------------
# 2. Construct hypergraph-style grant--PI incidence table
# -----------------------------
if "Core Project Number" not in combined_df.columns:
    combined_df["Core Project Number"] = combined_df["Project Number"].apply(
        extract_core_project_number
    )

hyperedge_records = []

for _, row in combined_df.iterrows():

    core_project = row["Core Project Number"]

    pis = []

    if pd.notna(row["Contact PI / Project Leader"]):
        pis.extend(process_pi_names(row["Contact PI / Project Leader"]))

    pis.extend(process_pi_names(row["Other PI or Project Leader(s)"]))

    pis = sorted(set(pis))

    for pi in pis:
        hyperedge_records.append((core_project, pi))

grant_pi_incidence_df = pd.DataFrame(
    hyperedge_records,
    columns=["core_project", "PI"]
)

# Remove repeated grant-year records
grant_pi_incidence_df = grant_pi_incidence_df.drop_duplicates(
    subset=["core_project", "PI"]
)

grant_pi_incidence_df.to_csv(
    "grant_pi_incidence_table.csv",
    index=False
)


# -----------------------------
# 3. Hyperedge size distribution
# -----------------------------
hyperedge_size_df = (
    grant_pi_incidence_df
    .groupby("core_project")
    .size()
    .reset_index(name="hyperedge_size")
)

hyperedge_counts = (
    hyperedge_size_df["hyperedge_size"]
    .value_counts()
    .sort_index()
)

hyperedge_percent = (
    100 * hyperedge_counts / hyperedge_counts.sum()
)

print("\nHyperedge size distribution (counts):")
print(hyperedge_counts)

print("\nHyperedge size distribution (%):")
print(hyperedge_percent.round(2))


# Remove singleton projects from plot
hyperedge_size_df_plot = hyperedge_size_df[
    hyperedge_size_df["hyperedge_size"] >= 2
].copy()

hyperedge_counts_plot = (
    hyperedge_size_df_plot["hyperedge_size"]
    .value_counts()
    .sort_index()
)

hyperedge_percent_plot = (
    100 * hyperedge_counts_plot / hyperedge_counts_plot.sum()
)

print("\nHyperedge size distribution after removing size 1 (counts):")
print(hyperedge_counts_plot)

print("\nHyperedge size distribution after removing size 1 (%):")
print(hyperedge_percent_plot.round(2))


# -----------------------------
# 4. Histogram of hyperedge sizes
# -----------------------------
plt.figure(figsize=(8, 5))

plt.hist(
    hyperedge_size_df_plot["hyperedge_size"],
    bins=np.arange(
        1.5,
        hyperedge_size_df_plot["hyperedge_size"].max() + 1.5,
        1
    ),
    edgecolor="black"
)

plt.xticks(
    range(
        2,
        hyperedge_size_df_plot["hyperedge_size"].max() + 1
    )
)

plt.xlabel("Hyperedge size: number of unique PIs in a core project")
plt.ylabel("Number of core projects")
plt.title("Distribution of Hyperedge Sizes")

plt.tight_layout()
plt.savefig("hist_hyperedge_sizes.png", dpi=300)
plt.show()
