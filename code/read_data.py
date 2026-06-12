import pandas as pd
import re


# ---- Read dataset ----
combined_df = pd.read_csv("combined_df.csv", low_memory=False)


# ---- Core project number (combine projects) ----
def extract_core_project_number(project_number):
    if pd.isna(project_number):
        return None

    project_number = str(project_number).strip()

    # Remove support year suffix, e.g. -01, -02, -05S1
    project_number = re.sub(r"-\d{2}.*$", "", project_number)

    # Remove leading application type digit, e.g. 1R01..., 7R01..., 3U01...
    project_number = re.sub(r"^\d", "", project_number)

    return project_number


combined_df["Project Number"] = combined_df["Project Number"].astype(str).str.strip()

combined_df["Core Project Number"] = combined_df["Project Number"].apply(
    extract_core_project_number
)

# ---- Project counts ----
n_grant_records = combined_df["Project Number"].nunique()
n_core_projects = combined_df["Core Project Number"].nunique()

print("Number of unique grant records:", n_grant_records)
print("Number of unique core projects:", n_core_projects)

# Reduce to one row per unique grant record
combined_df_unique_records = combined_df.drop_duplicates(
    subset="Project Number",
    keep="first"
).copy()

# Reduce to one row per underlying core project
combined_df_unique_projects = combined_df.drop_duplicates(
    subset="Core Project Number",
    keep="first"
).copy()

print("Unique grant records:", combined_df_unique_records["Project Number"].nunique())
print("Unique core projects:", combined_df_unique_projects["Core Project Number"].nunique())

# ---- Co-PI counts ----
other_pis = combined_df["Other PI or Project Leader(s)"].fillna("").astype(str)
total_co_pis = (other_pis.ne("") * (other_pis.str.count(";") + 1)).sum()
print("Total number of co-PIs in the dataset:", int(total_co_pis))

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


