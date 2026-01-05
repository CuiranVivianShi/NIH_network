import pandas as pd
import re  # Regular expressions module

data_with_correct_header = pd.read_csv("SearchResult_Export_17Jun2024_012813.csv", skiprows=4)
data_with_correct_header2 = pd.read_csv("SearchResult_Export_17Jun2024_012829.csv", skiprows=4)
data_with_correct_header3 = pd.read_csv("SearchResult_Export_17Jun2024_012845.csv", skiprows=4)

data_with_correct_header4 = pd.read_csv("SearchResult_Export_17Jun2024_012905.csv", skiprows=4)
data_with_correct_header5 = pd.read_csv("SearchResult_Export_17Jun2024_012916.csv", skiprows=4)
data_with_correct_header6 = pd.read_csv("SearchResult_Export_17Jun2024_012927.csv", skiprows=4)
data_with_correct_header7 = pd.read_csv("SearchResult_Export_17Jun2024_012938.csv", skiprows=4)
data_with_correct_header8 = pd.read_csv("SearchResult_Export_17Jun2024_012949.csv", skiprows=4)
data_with_correct_header9 = pd.read_csv("SearchResult_Export_17Jun2024_012959.csv", skiprows=4)
data_with_correct_header10 = pd.read_csv("SearchResult_Export_17Jun2024_013009.csv", skiprows=4)
data_with_correct_header11 = pd.read_csv("SearchResult_Export_17Jun2024_013019.csv", skiprows=4)
data_with_correct_header12 = pd.read_csv("SearchResult_Export_17Jun2024_013028.csv", skiprows=4)
data_with_correct_header13 = pd.read_csv("SearchResult_Export_17Jun2024_013109.csv", skiprows=4)
data_with_correct_header14 = pd.read_csv("SearchResult_Export_17Jun2024_013122.csv", skiprows=4)
data_with_correct_header15 = pd.read_csv("SearchResult_Export_17Jun2024_013132.csv", skiprows=4)
data_with_correct_header16 = pd.read_csv("SearchResult_Export_17Jun2024_013142.csv", skiprows=4)
data_with_correct_header17 = pd.read_csv("SearchResult_Export_17Jun2024_013150.csv", skiprows=4)
data_with_correct_header18 = pd.read_csv("SearchResult_Export_17Jun2024_013200.csv", skiprows=4)
data_with_correct_header19 = pd.read_csv("SearchResult_Export_17Jun2024_013209.csv", skiprows=4)
data_with_correct_header20 = pd.read_csv("SearchResult_Export_17Jun2024_013219.csv", skiprows=4)
data_with_correct_header21 = pd.read_csv("SearchResult_Export_17Jun2024_013228.csv", skiprows=4)

data_with_correct_header22 = pd.read_csv("SearchResult_Export_17Jun2024_013237.csv", skiprows=4)
data_with_correct_header23 = pd.read_csv("SearchResult_Export_17Jun2024_013246.csv", skiprows=4)
data_with_correct_header24 = pd.read_csv("SearchResult_Export_17Jun2024_013257.csv", skiprows=4)
data_with_correct_header25 = pd.read_csv("SearchResult_Export_17Jun2024_013306.csv", skiprows=4)
data_with_correct_header26 = pd.read_csv("SearchResult_Export_17Jun2024_013316.csv", skiprows=4)
data_with_correct_header27 = pd.read_csv("SearchResult_Export_17Jun2024_013335.csv", skiprows=4)
data_with_correct_header28 = pd.read_csv("SearchResult_Export_17Jun2024_013343.csv", skiprows=4)
data_with_correct_header29 = pd.read_csv("SearchResult_Export_17Jun2024_013352.csv", skiprows=4)
data_with_correct_header30 = pd.read_csv("SearchResult_Export_17Jun2024_013402.csv", skiprows=4)
data_with_correct_header31 = pd.read_csv("SearchResult_Export_17Jun2024_013412.csv", skiprows=4)
data_with_correct_header32 = pd.read_csv("SearchResult_Export_17Jun2024_013422.csv", skiprows=4)
data_with_correct_header33 = pd.read_csv("SearchResult_Export_17Jun2024_013430.csv", skiprows=4)
data_with_correct_header34 = pd.read_csv("SearchResult_Export_17Jun2024_013441.csv", skiprows=4)
data_with_correct_header35 = pd.read_csv("SearchResult_Export_17Jun2024_013453.csv", skiprows=4)
data_with_correct_header36 = pd.read_csv("SearchResult_Export_17Jun2024_013502.csv", skiprows=4)
data_with_correct_header37 = pd.read_csv("SearchResult_Export_17Jun2024_013511.csv", skiprows=4)
data_with_correct_header38 = pd.read_csv("SearchResult_Export_17Jun2024_013520.csv", skiprows=4)
data_with_correct_header39 = pd.read_csv("SearchResult_Export_17Jun2024_013529.csv", skiprows=4)
data_with_correct_header40 = pd.read_csv("SearchResult_Export_17Jun2024_013540.csv", skiprows=4)
data_with_correct_header41 = pd.read_csv("SearchResult_Export_17Jun2024_013548.csv", skiprows=4)


data_with_correct_header42 = pd.read_csv("SearchResult_Export_17Jun2024_013558.csv", skiprows=4)
data_with_correct_header43 = pd.read_csv("SearchResult_Export_17Jun2024_013609.csv", skiprows=4)
data_with_correct_header44 = pd.read_csv("SearchResult_Export_17Jun2024_013618.csv", skiprows=4)
data_with_correct_header45 = pd.read_csv("SearchResult_Export_17Jun2024_013628.csv", skiprows=4)
data_with_correct_header46 = pd.read_csv("SearchResult_Export_17Jun2024_013638.csv", skiprows=4)
data_with_correct_header47 = pd.read_csv("SearchResult_Export_17Jun2024_013649.csv", skiprows=4)
data_with_correct_header48 = pd.read_csv("SearchResult_Export_17Jun2024_013659.csv", skiprows=4)
data_with_correct_header49 = pd.read_csv("SearchResult_Export_17Jun2024_013710.csv", skiprows=4)
data_with_correct_header50 = pd.read_csv("SearchResult_Export_17Jun2024_013720.csv", skiprows=4)
data_with_correct_header51 = pd.read_csv("SearchResult_Export_17Jun2024_013735.csv", skiprows=4)


dfs = [data_with_correct_header, data_with_correct_header2, data_with_correct_header3,data_with_correct_header4,
       data_with_correct_header5, data_with_correct_header6, data_with_correct_header7, data_with_correct_header8,
       data_with_correct_header9, data_with_correct_header10, data_with_correct_header11, data_with_correct_header12,
       data_with_correct_header13, data_with_correct_header14, data_with_correct_header15, data_with_correct_header16,
       data_with_correct_header17, data_with_correct_header18, data_with_correct_header19, data_with_correct_header20,
       data_with_correct_header21, data_with_correct_header22, data_with_correct_header23, data_with_correct_header24,
       data_with_correct_header25, data_with_correct_header26, data_with_correct_header27, data_with_correct_header28,
       data_with_correct_header29, data_with_correct_header30, data_with_correct_header31, data_with_correct_header32,
       data_with_correct_header33, data_with_correct_header34, data_with_correct_header35, data_with_correct_header36,
       data_with_correct_header37, data_with_correct_header38, data_with_correct_header39, data_with_correct_header40,
       data_with_correct_header41, data_with_correct_header42, data_with_correct_header43, data_with_correct_header44,
       data_with_correct_header45, data_with_correct_header46, data_with_correct_header47, data_with_correct_header48,
       data_with_correct_header49, data_with_correct_header50, data_with_correct_header51]

combined_df = pd.concat(dfs, ignore_index=True)


def fetch_all_departments_NIHreporter(pis_to_rescrape, combined_df):
    department_info = []
    for pi in pis_to_rescrape:
        # Directly match the PI names without treating them as regex patterns
        matches = combined_df[(combined_df['Contact PI / Project Leader'].str.contains(pi, na=False, regex=False)) |
                              (combined_df['Other PI or Project Leader(s)'].str.contains(pi, na=False, regex=False))]
        if not matches.empty:
            # Select the entry with the latest fiscal year
            latest_entry = matches.sort_values('Fiscal Year', ascending=False).iloc[0]
            info = f"{latest_entry['Department']}, {latest_entry['Organization Name']}, {latest_entry['Organization City']}, " \
                   f"{latest_entry['Organization State']}, {latest_entry['Organization Zip']}, {latest_entry['Organization Country']}"
            department_info.append(info)
        else:
            # Append an empty string if no entries are found
            department_info.append("")
    return department_info


# Load the PI info where titles need to be rescraped
pi_info_df = pd.read_csv('PI_Info_Updated_Reporter12_largest_component.csv')
blank_indices = pi_info_df[pi_info_df['Title'].isna() | (pi_info_df['Title'] == "Error: 'NoneType' object has no attribute 'text'") | (pi_info_df['Title'] == "No valid affiliation found")].index.tolist()
pis_to_rescrape = pi_info_df.loc[blank_indices, 'PI Name'].tolist()

# Fetch new department information from NIH Reporter data
new_department_info = fetch_all_departments_NIHreporter(pis_to_rescrape, combined_df)


# Update the DataFrame
pi_info_df.loc[blank_indices, 'Title'] = new_department_info

# Save the updated DataFrame back to CSV
pi_info_df.to_csv('PI_Info_Updated_Reporter12_largest_component.csv', index=False)










import pandas as pd

pi_info_df = pd.read_csv('PI_Info_Updated_Reporter6.csv')
blank_title_pis = pi_info_df[pi_info_df['Title'].isna() | (pi_info_df['Title'] == '')]['PI Name'].tolist()
edges_df = pd.read_csv('edges_largest_component.csv')  # Replace 'path_to_edges_csv.csv' with the actual file path
pis_in_edges = set(edges_df['source'].tolist() + edges_df['target'].tolist())
count = sum(pi in pis_in_edges for pi in blank_title_pis)
print(f"Number of PIs with blank titles present in the edges: {count}")


#1. find patterns of collaboratin of PI
#2. get the fiscal year of each project



import pandas as pd

# Load the edges CSV file to access the source and target columns
edges_df = pd.read_csv('edges_largest_component.csv')

# Extract unique PI names from both 'source' and 'target' columns
unique_pis = pd.unique(edges_df[['source', 'target']].values.ravel('K'))

# Convert the numpy array back to a DataFrame for exporting to CSV
pi_names_df = pd.DataFrame(unique_pis, columns=['PI Names'])

# Save to CSV file
pi_names_df.to_csv('PI_name_largest_component.csv', index=False)




import pandas as pd

# Load data
nodes = pd.read_csv('PI_name_largest_component.csv')
edges = pd.read_csv('edges_largest_component.csv')
nodes = nodes['PI Names'].tolist()

# Create an empty adjacency matrix
adj_matrix = pd.DataFrame(0, index=nodes, columns=nodes)

# Populate the adjacency matrix
for index, row in edges.iterrows():
    start, end = row['source'], row['target']
    adj_matrix.loc[start, end] += 1
    adj_matrix.loc[end, start] += 1  # This line makes the graph undirected

# Save the adjacency matrix to a CSV file
adj_matrix.to_csv('adjacency_matrix_largest_component.csv')


import pandas as pd

# Load the data
pi_info_df = pd.read_csv('PI_Info_Updated_Reporter6.csv')
pi_names_df = pd.read_csv('PI_name_largest_component.csv')

# Ensure the names are treated as strings and are consistent in both dataframes
pi_info_df['PI Name'] = pi_info_df['PI Name'].str.strip()
pi_names_df['PI Names'] = pi_names_df['PI Names'].str.strip()

# Filter the dataframe to include only rows where the PI Name is in the largest component names list
filtered_df = pi_info_df[pi_info_df['PI Name'].isin(pi_names_df['PI Names'])]

# Save the filtered data to a new CSV file
filtered_df.to_csv('PI_Info_Updated_Reporter6_largest_component.csv', index=False)

print("Filtered data saved successfully.")

