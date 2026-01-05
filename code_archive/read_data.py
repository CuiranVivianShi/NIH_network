import pandas as pd

# Read the CSV file with the 7th row as the header, skipping the first 6 rows
data_with_correct_header = pd.read_csv("/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2006-2010/SearchResult_Export_2006.csv", skiprows=6)
data_with_correct_header2 = pd.read_csv("/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2006-2010/SearchResult_Export_2007.csv", skiprows=6)
data_with_correct_header3 = pd.read_csv("/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2006-2010/SearchResult_Export_2008.csv", skiprows=6)
data_with_correct_header4 = pd.read_csv("/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2006-2010/SearchResult_Export_2009.csv", skiprows=6)
data_with_correct_header5 = pd.read_csv("/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2006-2010/SearchResult_Export_2010.csv", skiprows=6)
data_with_correct_header6 = pd.read_csv("/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2011-2015/SearchResult_Export_2011.csv", skiprows=6)
data_with_correct_header7 = pd.read_csv("/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2011-2015/SearchResult_Export_2012.csv", skiprows=6)
data_with_correct_header8 = pd.read_csv("/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2011-2015/SearchResult_Export_2013.csv", skiprows=6)
data_with_correct_header9 = pd.read_csv("/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2011-2015/SearchResult_Export_2014.csv", skiprows=6)
data_with_correct_header10 = pd.read_csv("/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2011-2015/SearchResult_Export_2015.csv", skiprows=6)

dfs = [data_with_correct_header, data_with_correct_header2, data_with_correct_header3, data_with_correct_header4,
       data_with_correct_header5, data_with_correct_header6, data_with_correct_header7, data_with_correct_header8, data_with_correct_header9, data_with_correct_header10]

# Concatenate dataframes
combined_df = pd.concat(dfs, ignore_index=True)
#print(combined_df.head(), combined_df.shape)

#combined_df.to_csv('combined_df.csv', index=False)

#Import the dataframe
combined_df = pd.read_csv('combined_df.csv', low_memory=False)
#combined_df = pd.read_csv("SearchResult_Export_2023.csv", skiprows=6)
#data_with_correct_header = pd.read_csv("SearchResult_Export_2006.csv", skiprows=6)
#data_with_correct_header2 = pd.read_csv("SearchResult_Export_2007.csv", skiprows=6)
#data_with_correct_header3 = pd.read_csv("SearchResult_Export_2008.csv", skiprows=6)
#dfs = [data_with_correct_header, data_with_correct_header2, data_with_correct_header3]
#combined_df = pd.concat(dfs, ignore_index=True)

# Number of projects
unique_project_numbers = combined_df['Application ID'].nunique()
print(unique_project_numbers) # 86787 correct

# Combine the same project from different years (i.e., first 12 char the same)
data_modified = combined_df.copy()
data_modified['Project Number'] = data_modified['Project Number'].str.slice(0, 12)
data_modified.drop_duplicates(subset='Project Number', inplace=True, keep='first')
unique_project_numbers_modified = data_modified['Application ID'].nunique()
print(unique_project_numbers_modified) #47805

# Count the total number of co-PIs in the dataset
total_co_pis = data_modified['Other PI or Project Leader(s)'].str.count(';') + 1
print("Total number of co-PIs in the dataset:", total_co_pis.sum()) #61176

# Count the total number of PIs in the dataset (Same as project numbers)
print(unique_project_numbers_modified) #47805

# Count total number of distinct PI and Co-PIs
# Extract and clean PI names
pis = combined_df['Contact PI / Project Leader'].dropna().str.split(';').apply(lambda x: [pi.strip() for pi in x])
pis_list = [pi for sublist in pis for pi in sublist]
# Extract and clean Co-PI names
co_pis = combined_df['Other PI or Project Leader(s)'].dropna().str.split(';').apply(lambda x: [co_pi.strip() for co_pi in x])
co_pis_list = [co_pi for sublist in co_pis for co_pi in sublist]
# Combine all PI and Co-PI names in a set to remove duplicates
all_unique_pis = set(pis_list + co_pis_list)
# Count the total number of unique PIs and Co-PIs
total_unique_pis_and_co_pis = len(all_unique_pis)
print("Total number of unique PIs and Co-PIs combined:", total_unique_pis_and_co_pis) #30133 revised: 30127



def get_combined_df():

    return combined_df

def get_unique_pis():

    return all_unique_pis



# Convert the set of PI names into a DataFrame
pi_df = pd.DataFrame(list(all_unique_pis), columns=['PI Names'])

# Save the DataFrame to a CSV file
#pi_df.to_csv('all_unique_pis.csv', index=False)

#print("PI names have been saved to 'all_unique_pis.csv'")