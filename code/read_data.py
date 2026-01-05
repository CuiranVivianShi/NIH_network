import pandas as pd

# ---- Read datasets ----
data_2006 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2006-2010/SearchResult_Export_2006.csv",
    skiprows=6
)
data_2007 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2006-2010/SearchResult_Export_2007.csv",
    skiprows=6
)
data_2008 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2006-2010/SearchResult_Export_2008.csv",
    skiprows=6
)
data_2009 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2006-2010/SearchResult_Export_2009.csv",
    skiprows=6
)
data_2010 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2006-2010/SearchResult_Export_2010.csv",
    skiprows=6
)

data_2011 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2011-2015/SearchResult_Export_2011.csv",
    skiprows=6
)
data_2012 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2011-2015/SearchResult_Export_2012.csv",
    skiprows=6
)
data_2013 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2011-2015/SearchResult_Export_2013.csv",
    skiprows=6
)
data_2014 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2011-2015/SearchResult_Export_2014.csv",
    skiprows=6
)
data_2015 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2011-2015/SearchResult_Export_2015.csv",
    skiprows=6
)

data_2016 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2016-2020/SearchResult_Export_2016.csv",
    skiprows=6
)
data_2017 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2016-2020/SearchResult_Export_2017.csv",
    skiprows=6
)
data_2018 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2016-2020/SearchResult_Export_2018.csv",
    skiprows=6
)
data_2019 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2016-2020/SearchResult_Export_2019.csv",
    skiprows=6
)
data_2020 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2016-2020/SearchResult_Export_2020.csv",
    skiprows=6
)

data_2021 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2021-2023/SearchResult_Export_2021.csv",
    skiprows=6
)
data_2022 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2021-2023/SearchResult_Export_2022.csv",
    skiprows=6
)
data_2023 = pd.read_csv(
    "/Users/shicuiran/PycharmProjects/NIH_network/data_stratified_by_5_year/5year_2021-2023/SearchResult_Export_2023.csv",
    skiprows=6
)

# ---- Combine datasets ----
dfs = [
    data_2006, data_2007, data_2008, data_2009, data_2010,
    data_2011, data_2012, data_2013, data_2014, data_2015,
    data_2016, data_2017, data_2018, data_2019, data_2020,
    data_2021, data_2022, data_2023
]

combined_df = pd.concat(dfs, ignore_index=True)

# ---- Project counts ----
n_projects = combined_df["Project Number"].nunique()
print(n_projects)

# reduce to one row per unique Project Number
combined_df_unique = combined_df.drop_duplicates(subset='Project Number', keep='first')
combined_df_unique['Project Number'].nunique() #86,743

# ---- Co-PI counts ----
other_pis = combined_df["Other PI or Project Leader(s)"].fillna("").astype(str)
total_co_pis = (other_pis.ne("") * (other_pis.str.count(";") + 1)).sum()
print("Total number of co-PIs in the dataset:", int(total_co_pis))  # 61176

# ---- Unique PI names ----
pis = combined_df["Contact PI / Project Leader"].dropna().str.split(";")
pis_list = [p.strip() for sub in pis for p in sub]

co_pis = combined_df["Other PI or Project Leader(s)"].dropna().str.split(";")
co_pis_list = [p.strip() for sub in co_pis for p in sub]

all_unique_pis = set(pis_list + co_pis_list)
print("Total number of unique PIs and Co-PIs combined:", len(all_unique_pis))

pi_df = pd.DataFrame(sorted(all_unique_pis), columns=["PI Names"])
# pi_df.to_csv("all_unique_pis.csv", index=False)

def get_combined_df():
    return combined_df.copy()

def get_unique_pis():
    return set(all_unique_pis)
