import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.gridspec import GridSpec

# ============================================================
# 0) Setup
# ============================================================
OUTDIR = "summary_statistics_manuscript"
os.makedirs(OUTDIR, exist_ok=True)

primary_blue = "#2b6cb0"
secondary_orange = "#e6550d"
light_gray = "#bdbdbd"
dark_gray = "#4a4a4a"

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

comma_fmt = FuncFormatter(lambda x, _: f"{int(x):,}")

# ============================================================
# 1) Load and prepare data
# ============================================================
combined_df = pd.read_csv("combined_df.csv", low_memory=False)
all_unique_pis = pd.read_csv("nodes.csv")

# Unique project at grant-year level
df = combined_df.drop_duplicates(subset="Project Number", keep="first").copy()

# Clean columns
df["Project Number"] = df["Project Number"].astype(str).str.strip()
df["Organization Name"] = df["Organization Name"].fillna("Unknown").astype(str).str.strip()
df["Department"] = df["Department"].fillna("Unknown Department").astype(str).str.strip()
df["Organization Country"] = df["Organization Country"].fillna("Unknown").astype(str).str.strip()
df["Total Cost"] = pd.to_numeric(df["Total Cost"], errors="coerce")

total_grants = df["Project Number"].nunique()
total_pis = all_unique_pis["PI Names"].nunique()
n_orgs = df["Organization Name"].nunique()
n_countries = df["Organization Country"].nunique()

# Clean fiscal year
df["Fiscal Year"] = df["Fiscal Year"].astype(str)
df = df[df["Fiscal Year"].str.isdigit()].copy()
df["Fiscal Year"] = df["Fiscal Year"].astype(int)

# Funding-positive subset
funding_df = df[df["Total Cost"].notna() & (df["Total Cost"] > 0)].copy()

# ============================================================
# 2) Identify the NIH Institute/Center (IC) column
# ============================================================
possible_ic_cols = [
    "Administering IC",
    "Agency IC Admin",
    "IC Name",
    "Funding IC",
    "Institute/Center"
]

ic_col = None
for col in possible_ic_cols:
    if col in df.columns:
        ic_col = col
        break

n_ic = df[ic_col].nunique() if ic_col is not None else np.nan

# ============================================================
# 3) Summary metrics for manuscript Table 1
# ============================================================
fy_min = df["Fiscal Year"].min()
fy_max = df["Fiscal Year"].max()
total_funding = funding_df["Total Cost"].sum()
median_funding = funding_df["Total Cost"].median()
q1_funding = funding_df["Total Cost"].quantile(0.25)
q3_funding = funding_df["Total Cost"].quantile(0.75)

table1_df = pd.DataFrame({
    "Characteristic": [
        "Total unique grants",
        "Total unique PIs",
        "NIH funding Institutes/Centers",
        "Recipient organizations",
        "Recipient countries",
        "Fiscal year range",
        "Total funding",
        "Median award funding",
        "Award funding, IQR"
    ],
    "Value": [
        f"{total_grants:,}",
        f"{total_pis:,}",
        f"{int(n_ic):,}" if pd.notna(n_ic) else "NA",
        f"{n_orgs:,}",
        f"{n_countries:,}",
        f"{fy_min}–{fy_max}",
        f"${total_funding:,.0f}",
        f"${median_funding:,.0f}",
        f"${q1_funding:,.0f}–${q3_funding:,.0f}"
    ]
})

table1_df.to_csv(f"{OUTDIR}/table1_dataset_overview.csv", index=False)

# ============================================================
# 4) Fiscal-year summary for Figure 1A
# ============================================================
fy_summary = (
    df.groupby("Fiscal Year", dropna=False)
      .agg(
          n_grants=("Project Number", "nunique"),
          total_funding=("Total Cost", "sum")
      )
      .reset_index()
      .sort_values("Fiscal Year")
)

# ============================================================
# 5) Organization funding summary
# ============================================================
org_funding = (
    df.groupby("Organization Name", dropna=False)
      .agg(
          total_funding=("Total Cost", "sum"),
          n_grants=("Project Number", "nunique")
      )
      .reset_index()
      .sort_values("total_funding", ascending=False)
)

df_top_org = org_funding.head(10).copy()

# Shorter labels for panel C
df_top_org["Org Clean"] = df_top_org["Organization Name"].astype(str).str.strip().str.lower()
name_map = {
    "johns hopkins university": "Johns Hopkins University",
    "university of california, san francisco": "UCSF",
    "university of pittsburgh at pittsburgh": "University of Pittsburgh",
    "columbia university health sciences": "Columbia Health Sciences",
    "university of michigan at ann arbor": "Univ. of Michigan",
    "massachusetts general hospital": "Mass General Hospital",
    "university of pennsylvania": "University of Pennsylvania",
    "yale university": "Yale",
    "university of california, san diego": "UC San Diego",
    "washington university": "Washington University"
}
df_top_org["Org Label"] = df_top_org["Org Clean"].map(name_map).fillna(df_top_org["Organization Name"])
df_top_org_plot = df_top_org.sort_values("total_funding", ascending=True).copy()

# ============================================================
# 6) Helper: save Table 1 as figure
# ============================================================
def save_table_as_figure(df_table, title, filename, col_widths=(0.68, 0.32), fontsize=12):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axis("off")

    tbl = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        colLoc="left",
        cellLoc="left",
        colWidths=list(col_widths),
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.55)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")

    ax.plot([0.02, 0.98], [0.92, 0.92], color="black", lw=0.8, transform=ax.transAxes, clip_on=False)
    ax.plot([0.02, 0.98], [0.11, 0.11], color="black", lw=0.8, transform=ax.transAxes, clip_on=False)

    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

save_table_as_figure(
    table1_df,
    "Table 1. Overview of the multi-PI NIH R01-equivalent grant dataset (2006–2023)",
    f"{OUTDIR}/table1_dataset_overview.png"
)

# ============================================================
# 7) Composite Figure 1
# ============================================================
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

# ------------------------------------------------------------
# Panel A: Annual grant volume and funding
# ------------------------------------------------------------
axA = fig.add_subplot(gs[0, 0])

years = fy_summary["Fiscal Year"].to_numpy()

bars = axA.bar(
    years,
    fy_summary["n_grants"],
    color=primary_blue,
    alpha=0.85,
    edgecolor="none",
    label="Number of grants"
)
axA.set_ylabel("Number of grants")
axA.set_xlabel("Fiscal year")
axA.yaxis.set_major_formatter(comma_fmt)

# force integer year display
axA.set_xticks(years)
axA.xaxis.set_major_locator(MaxNLocator(integer=True))
axA.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))
axA.tick_params(axis="x", rotation=45)

# add number of grants above bars
ymax_grants = fy_summary["n_grants"].max()
for x, y in zip(years, fy_summary["n_grants"]):
    axA.text(
        x,
        y + ymax_grants * 0.06,
        f"{y:,}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#d62728"
    )
axA2 = axA.twinx()
axA2.plot(
    years,
    fy_summary["total_funding"],
    color=secondary_orange,
    linewidth=2.2,
    marker="o",
    markersize=4,
    label="Total funding"
)
axA2.set_ylabel("Total funding ($)")
axA2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{x/1e9:.1f}B"))

axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)
axA2.spines["top"].set_visible(False)
axA.grid(axis="y", color=light_gray, alpha=0.25)

# give a little headroom for labels
axA.set_ylim(0, ymax_grants * 1.12)

lines1, labels1 = axA.get_legend_handles_labels()
lines2, labels2 = axA2.get_legend_handles_labels()
axA.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

axA.set_title("(a) Annual grant volume and funding")

# ------------------------------------------------------------
# Panel B: Funding distribution with log-scale frequency only
# ------------------------------------------------------------
axB = fig.add_subplot(gs[0, 1])

axB.hist(
    funding_df["Total Cost"],
    bins=50,
    color=primary_blue,
    edgecolor="white",
    linewidth=0.4,
    alpha=0.95
)
axB.set_yscale("log")
axB.set_xlabel("Total cost per award ($)")
axB.set_ylabel("Frequency")
axB.set_title("(b) Distribution of award funding")

axB.xaxis.set_major_formatter(comma_fmt)
axB.tick_params(axis="x", labelsize=9)
axB.tick_params(axis="y", labelsize=9)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.grid(axis="y", color=light_gray, alpha=0.25)

# ------------------------------------------------------------
# Panel C: Top 10 organizations by total funding
# ------------------------------------------------------------
axC = fig.add_subplot(gs[1, 0])

bars = axC.barh(
    df_top_org_plot["Org Label"],
    df_top_org_plot["total_funding"],
    color=primary_blue,
    edgecolor="none"
)

xmax = df_top_org_plot["total_funding"].max()
for bar, grants in zip(bars, df_top_org_plot["n_grants"]):
    axC.text(
        bar.get_width() + xmax * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"n={grants:,}",
        va="center",
        ha="left",
        fontsize=9
    )

axC.set_xlabel("Total funding ($)")
axC.set_ylabel("")
axC.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{x/1e9:.1f}B"))
axC.set_xlim(0, xmax * 1.2)
axC.set_title("(c) Top 10 organizations by total funding")
axC.spines["top"].set_visible(False)
axC.spines["right"].set_visible(False)
axC.grid(axis="x", color=light_gray, alpha=0.25)

# ------------------------------------------------------------
# Panel D: Dataset overview table
# ------------------------------------------------------------
axD = fig.add_subplot(gs[1, 1])
axD.axis("off")
axD.set_title(
    "(d) Dataset overview",
    fontsize=12,
    pad=8
)

table = axD.table(
    cellText=table1_df.values,
    colLabels=table1_df.columns,
    colLoc="left",
    cellLoc="left",
    colWidths=[0.64, 0.28],
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.45)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("white")
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#f2f2f2")

# manuscript-like top/bottom rules
axD.plot([0.03, 0.97], [0.92, 0.92], color="black", lw=0.8, transform=axD.transAxes, clip_on=False)
axD.plot([0.03, 0.97], [0.10, 0.10], color="black", lw=0.8, transform=axD.transAxes, clip_on=False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    f"{OUTDIR}/figure1_summary_statistics_composite_revised.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)
plt.close()

print("Saved files:")
print(f"  {OUTDIR}/table1_dataset_overview.csv")
print(f"  {OUTDIR}/table1_dataset_overview.png")
print(f"  {OUTDIR}/figure1_summary_statistics_composite_revised.png")
