import pandas as pd
import re  # Regular expressions module

data_with_correct_header = pd.read_csv("SearchResult_Export_2000-2012.csv", skiprows=6)
data_with_correct_header2 = pd.read_csv("SearchResult_Export_2013-2015.csv", skiprows=6)
data_with_correct_header3 = pd.read_csv("SearchResult_Export_2016-2017.csv", skiprows=6)
data_with_correct_header4 = pd.read_csv("SearchResult_Export_2018.csv", skiprows=6)
data_with_correct_header5 = pd.read_csv("SearchResult_Export_2019.csv", skiprows=6)
data_with_correct_header6 = pd.read_csv("SearchResult_Export_2020.csv", skiprows=6)
data_with_correct_header7 = pd.read_csv("SearchResult_Export_2021.csv", skiprows=6)
data_with_correct_header8 = pd.read_csv("SearchResult_Export_2022.csv", skiprows=6)
data_with_correct_header9 = pd.read_csv("SearchResult_Export_2023.csv", skiprows=6)

dfs = [data_with_correct_header, data_with_correct_header2, data_with_correct_header3, data_with_correct_header4,
       data_with_correct_header5, data_with_correct_header6, data_with_correct_header7, data_with_correct_header8, data_with_correct_header9]

combined_df = pd.concat(dfs, ignore_index=True)


def fetch_all_departments_NIHreporter(pis_to_rescrape, combined_df):
    department_info = []
    for pi in pis_to_rescrape:
        # Directly match the PI names without treating them as regex patterns
        matches = combined_df[combined_df['Contact PI / Project Leader'].str.contains(pi, na=False, regex=False)]
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
pi_info_df = pd.read_csv('PI_Info_Updated_Reporter8_largest_component.csv')
blank_indices = pi_info_df[pi_info_df['Title'].isna() | (pi_info_df['Title'] == "Error: 'NoneType' object has no attribute 'text'") | (pi_info_df['Title'] == "No valid affiliation found")].index.tolist()
pis_to_rescrape = pi_info_df.loc[blank_indices, 'PI Name'].tolist()

# Fetch new department information from NIH Reporter data
new_department_info = fetch_all_departments_NIHreporter(pis_to_rescrape, combined_df)


# Update the DataFrame
pi_info_df.loc[blank_indices, 'Title'] = new_department_info

# Save the updated DataFrame back to CSV
pi_info_df.to_csv('PI_Info_Updated_Reporter9_largest_component.csv', index=False)


