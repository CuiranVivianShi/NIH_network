import pandas as pd
from itertools import combinations
import networkx as nx

from read_data import get_combined_df, get_unique_pis

combined_df = get_combined_df()
all_unique_pis = get_unique_pis()
nodes = list(all_unique_pis)

def process_pi_names(pi_str):
    if pd.isna(pi_str):
        return []
    return [name.strip() for name in str(pi_str).split(";") if name.strip()]

# ---- Build edges from co-participation ----
edges = []
for _, row in combined_df.iterrows():
    pis = [row["Contact PI / Project Leader"].strip()] if pd.notna(row["Contact PI / Project Leader"]) else []
    pis.extend(process_pi_names(row["Other PI or Project Leader(s)"]))
    edges.extend(combinations(pis, 2))

print("Total edges:", len(edges))
for e in edges[:10]:
    print(e)

# ---- Adjacency matrix (counts during construction) ----
adj_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
for start, end in edges:
    adj_matrix.loc[start, end] += 1
    adj_matrix.loc[end, start] += 1

print(adj_matrix.head())

# ---- Save full network outputs ----
nodes_df = pd.DataFrame(nodes, columns=["PI Names"])
edges_df = pd.DataFrame(edges, columns=["source", "target"])

edges_df.to_csv("edges.csv", index=False)
nodes_df.to_csv("nodes.csv", index=False)
adj_matrix.to_csv("adjacent.csv", index=True)

# ---- Largest connected component (LCC) ----
A = adj_matrix.to_numpy()
G = nx.from_numpy_array(A)

connected_components = list(nx.connected_components(G))
component_sizes = [len(c) for c in connected_components]
print("Number of components:", len(component_sizes))
print("Largest component size:", max(component_sizes))

largest_cc = max(connected_components, key=len)
subgraph = G.subgraph(largest_cc)

edges_lcc_idx = list(subgraph.edges())
edges_lcc_named = [(nodes[i], nodes[j]) for i, j in edges_lcc_idx]
edges_lcc_df = pd.DataFrame(edges_lcc_named, columns=["source", "target"])
edges_lcc_df.to_csv("edges_largest_component.csv", index=False)

# ---- Nodes in LCC ----
unique_pis_lcc = set()
for u, v in edges_lcc_named:
    unique_pis_lcc.add(u)
    unique_pis_lcc.add(v)

pis_lcc_df = pd.DataFrame(sorted(unique_pis_lcc), columns=["PI Names"])
pis_lcc_df.to_csv("nodes_largest_component.csv", index=False)

# ---- Adjacency matrix for LCC ----
nodes_lcc = sorted(unique_pis_lcc)
adj_matrix_lcc = pd.DataFrame(0, index=nodes_lcc, columns=nodes_lcc)

for start, end in edges_lcc_named:
    adj_matrix_lcc.loc[start, end] += 1
    adj_matrix_lcc.loc[end, start] += 1

adj_matrix_lcc.to_csv("adjacent_matrix_largest_component.csv", index=True)
