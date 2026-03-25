from community_detection import all_matches
import sent2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy
import nltk
from nltk.corpus import stopwords
from scipy.spatial import distance
import pandas as pd
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy
import nltk
from nltk.corpus import stopwords
import re
import certifi
print(certifi.where())
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')


combined_df = pd.read_csv('combined_df.csv', low_memory=False)

# Convert to string type
combined_df['Project Title'] = combined_df['Project Title'].astype(str)
combined_df['Project Abstract'] = combined_df['Project Abstract'].astype(str)

# Drop duplicates based on both 'Project Title' and 'Project Abstract'.
# This will return the first record appeared, usually the first fiscal year
combined_df = combined_df.drop_duplicates(subset=['Project Title', 'Project Abstract'])

# Add fiscal year
project_text = combined_df[['Project Title', 'Project Abstract', 'Fiscal Year']].copy()

# Combine 'Project Title' and 'Project Abstract' into a new 'Project Text' column
project_text["Project Text"] = project_text['Project Title'] + " " + project_text['Project Abstract']


def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'\b(ABSTRACT|abstract|Project Summary)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'DESCRIPTION \(provided by applicant\):', '', text, flags=re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text) # Remove extra spaces
    text = re.sub(r'\S*@\S*\s?', '', text) # Remove emails
    text = re.sub('\'', '', text) # Remove apostrophes
    text = re.sub(r'[^a-zA-Z ]', '', text) # Remove non-alphabet characters except spaces
    text = text.lower()

    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(filtered_tokens)

project_text['cleaned_text'] = project_text['Project Text'].apply(preprocess_text)

# Sample for model training
project_text_sample = project_text.sample(n=5000, random_state=42)
project_text_full = project_text


"""
Load BioSentVec model
"""
model_path = "/Users/shicuiran/PycharmProjects/NIH_network/topic_modeling/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
model = sent2vec.Sent2vecModel()
try:
    model.load_model(model_path)
except Exception as e:
    print(e)
print('model successfully loaded')

# Retrieve sentence vectors
project_text_sample['embeddings'] = project_text_sample['cleaned_text'].apply(model.embed_sentence)
project_text_full['embeddings'] = project_text_full['cleaned_text'].apply(model.embed_sentence)

# Flatten the embeddings by removing the outer list
project_text_sample['embeddings'] = project_text_sample['embeddings'].apply(lambda x: x[0] if len(x) > 0 else None)
project_text_full['embeddings'] = project_text_full['embeddings'].apply(lambda x: x[0] if len(x) > 0 else None)

# Convert the list of embeddings to a NumPy array
embeddings_array = np.vstack(project_text_sample['embeddings'])
embeddings_array_full = np.vstack(project_text_full['embeddings'])




"""
Add Time 
"""
# Save original indexes
original_indexes = project_text_sample.index

original_indexes_full = project_text_full.index

import pandas as pd

# Convert 'Fiscal Year' to string and filter out non-numeric entries
project_text_sample['Fiscal Year'] = project_text_sample['Fiscal Year'].astype(str)
project_text_sample = project_text_sample[project_text_sample['Fiscal Year'].str.isdigit()]

project_text_full['Fiscal Year'] = project_text_full['Fiscal Year'].astype(str)
project_text_full = project_text_full[project_text_full['Fiscal Year'].str.isdigit()]

# Append the standard date and time to create a full datetime string
project_text_sample['Full Fiscal DateTime'] = project_text_sample['Fiscal Year'].apply(lambda x: f"{x}-01-01 00:00:00")

project_text_full['Full Fiscal DateTime'] = project_text_full['Fiscal Year'].apply(lambda x: f"{x}-01-01 00:00:00")

# Convert the string to actual datetime objects
project_text_sample['Full Fiscal DateTime'] = pd.to_datetime(project_text_sample['Full Fiscal DateTime'], errors='coerce')

project_text_full['Full Fiscal DateTime'] = pd.to_datetime(project_text_full['Full Fiscal DateTime'], errors='coerce')

# Check for any conversion failures
print(project_text_sample['Full Fiscal DateTime'].isnull().sum())

print(project_text_full['Full Fiscal DateTime'].isnull().sum())

import numpy as np

# Assuming embeddings_array originally corresponded to the full project_text_sample
# Update embeddings_array to match the filtered DataFrame
filtered_indexes = [i for i, idx in enumerate(original_indexes) if idx in project_text_sample.index]
embeddings_array = embeddings_array[filtered_indexes]

filtered_indexes_full = [i for i, idx in enumerate(original_indexes_full) if idx in project_text_full.index]
embeddings_array_full = embeddings_array_full[filtered_indexes_full]


"""
BERTopic Model
"""
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired

# Create deterministic UMAP and HDBSCAN instances
umap_model = UMAP(n_neighbors=15, n_components=5, #15, 5
                 min_dist=0.05, metric='cosine',
                 random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=15,
                       metric='euclidean',
                       min_samples=5,
                       cluster_selection_method='eom',
                       prediction_data=True)

representation_model = KeyBERTInspired()
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    embedding_model=None,
    calculate_probabilities=True,
    verbose=True,
    nr_topics=21
)

# Fit BERTopic
topics, probabilities = topic_model.fit_transform(
    project_text_sample['cleaned_text'],
    embeddings_array
)

# Print topics to see results
print("Topics found:", topics)

from collections import Counter
# Count frequency of each topic
topic_counts = Counter(topics)
# Print sorted counts by topic ID
for topic, count in sorted(topic_counts.items()):
    print(f"Topic {topic}: {count}")

# Generate visualizations
topic_visualization = topic_model.visualize_topics()
barchart_visualization = topic_model.visualize_barchart()
heatmap = topic_model.visualize_heatmap()

# Save visualizations to PNG files
topic_visualization.write_image(
    "/Users/shicuiran/PycharmProjects/NIH_network/topic_model/topic_visualization.png",
    engine="kaleido",
    scale=4   # increases DPI without changing layout proportions
)

barchart_visualization.write_image(
    "/Users/shicuiran/PycharmProjects/NIH_network/topic_model/barchart_visualization.png",
    engine="kaleido",
    scale=4
)

heatmap.write_image(
    "/Users/shicuiran/PycharmProjects/NIH_network/topic_model/heatmap.png",
    engine="kaleido",
    scale=4
)

# Count the unique topics
unique_topics = np.unique(topics)
number_of_topics = len(unique_topics) - 1 if -1 in unique_topics else len(unique_topics)
print(f"Number of topics found: {number_of_topics}")


# Retrieve the top n words for each topic
topic_keywords = topic_model.get_topic_info()

# Print out the topics and their keywords
print(topic_keywords)

# Get all topics (excluding outlier topic if present)
all_topics = topic_model.get_topics()
if -1 in all_topics:
    del all_topics[-1]
# List of all topic IDs
topic_ids = list(all_topics.keys())


import plotly.express as px

# Generate the barchart
barchart_visualization = topic_model.visualize_barchart(topics=topic_ids, n_words=10)

# Use a qualitative color palette
distinct_colors = px.colors.qualitative.Dark24  # 24 distinct colors


# Apply colors (cycling if >20 topics)
for i, topic_trace in enumerate(barchart_visualization.data):
    topic_trace.marker.color = distinct_colors[i % len(distinct_colors)]

# Save the visualization to a PNG file

barchart_visualization.write_image(
    "/Users/shicuiran/PycharmProjects/NIH_network/topic_model/barchart_visualization_full.png",
    engine="kaleido",
    scale=4
)

## Genrate 1 topic and specify color
import matplotlib.pyplot as plt

# Assuming 'all_topics' is a dictionary containing topics and their words with associated scores
# Extracting data for Topic 0
topic_4_data = all_topics[4]  # [('word', score), ...]
words, scores = zip(*topic_4_data)  # Unpacking words and their scores

# Create a bar chart with adjusted dimensions
fig, ax = plt.subplots(figsize=(10, 10))  # Set figure dimensions
ax.barh(words, scores, color='#2ca02c')  # Placeholder blue color, replace '#007BFF' with the exact hex code
ax.set_xlabel('Scores')
ax.set_title('Topic 4',fontsize=15)
ax.invert_yaxis()  # Highest score at the top
ax.set_facecolor('white')  # Set the background to white
plt.box(False)  # Turn off the box frame around the plot

# Customizing the font and layout
ax.xaxis.set_tick_params(width=0)  # Hide x-axis ticks
ax.yaxis.set_tick_params(width=0)  # Hide y-axis ticks
plt.xticks(fontsize=15)  # Set x-tick label size
plt.yticks(fontsize=15)  # Set y-tick label size

# Remove all spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Save the plot as a PNG file
plt.savefig('/Users/shicuiran/PycharmProjects/NIH_network/topic_model/Topic_4_Word_Scores.png', bbox_inches='tight')





"""
Topic prediction
"""
cluster_1_matches = all_matches[0]
cluster_1_matches_title = cluster_1_matches[['Project Title']].copy()
cluster_1_matches_title['cleaned_text'] = cluster_1_matches_title['Project Title'].apply(preprocess_text)
cluster_1_matches_title['embeddings'] = cluster_1_matches_title['cleaned_text'].apply(model.embed_sentence)
cluster_1_matches_title['embeddings'] = cluster_1_matches_title['embeddings'].apply(lambda x: x[0] if len(x) > 0 else None)
cluster_1_embeddings_array = np.vstack(cluster_1_matches_title['embeddings'])
topics1, probabilities1 = topic_model.transform(cluster_1_matches_title['cleaned_text'], cluster_1_embeddings_array)
# Print topics to see results
print("Topics found:", topics1)

cluster_1_matches_title['predicted_topic'] = topics1
topic_counts1 = cluster_1_matches_title['predicted_topic'].value_counts()
print(topic_counts1)



import numpy as np
np.random.seed(42)
topic_results = []  # This will store results for each cluster

for i in range(19):  # Looping through clusters 0 to 18
    print(f"Processing Cluster {i}:")
    cluster_matches = all_matches[i]
    cluster_matches_text = cluster_matches[['Project Title','Project Abstract']].astype(str).copy().drop_duplicates()
    cluster_matches_text["Project Text"] = cluster_matches_text['Project Title'] + " " + cluster_matches_text['Project Abstract']
    cluster_matches_text['cleaned_text'] = cluster_matches_text["Project Text"].apply(preprocess_text)
    cluster_matches_text['embeddings'] = cluster_matches_text['cleaned_text'].apply(model.embed_sentence)
    cluster_matches_text['embeddings'] = cluster_matches_text['embeddings'].apply(
        lambda x: x[0] if len(x) > 0 else None)
    cluster_embeddings_array = np.vstack([e for e in cluster_matches_text['embeddings'] if e is not None])

    if cluster_embeddings_array.size > 0:  # Only proceed if there are embeddings
        topics, probabilities = topic_model.transform(cluster_matches_text['cleaned_text'], cluster_embeddings_array)

        cluster_matches_text['predicted_topic'] = topics
        topic_counts = cluster_matches_text['predicted_topic'].value_counts()
        print(topic_counts)

        topic_results.append({
            'cluster': i,
            'topics': topics,
            'probabilities': probabilities,
            'topic_counts': topic_counts
        })





"""
Donut Plots: Topic Distribution for Each Community
"""

# Write donut plots in a loop
import matplotlib.pyplot as plt
import plotly.express as px
import os
import numpy as np
import textwrap

# Create output directory if it doesn't exist
output_dir = '/Users/shicuiran/PycharmProjects/NIH_network/topic_model/cluster_plots'
os.makedirs(output_dir, exist_ok=True)

# Use Plotly's Dark24 qualitative color palette
distinct_colors = px.colors.qualitative.Dark24

# ---- Topic summary mapping ----
topic_summaries = {
    0: "Healthcare and Outcomes Research",
    1: "Cellular Biology and Functions",
    2: "Cardiovascular and Cognitive Health",
    3: "Neuroscience and Brain Disorders",
    4: "Cancer Immunotherapy",
    5: "Genomics and Genetic Variation",
    6: "Molecular and Cellular Biology",
    7: "Neurology and Pain Management",
    8: "Microbiome Research",
    9: "Medical Imaging Technologies",
    10: "Virology: HIV and Viral Infections",
    11: "Pharmacology and Drug Development",
    12: "Alzheimer’s Disease and Neurodegenerative Disease",
    13: "Infectious Disease in Developing Regions",
    14: "Vaccines and Immune Response",
    15: "Bone Health and Cancer",
    16: "Ophthalmology and Vision Sciences",
    17: "Liver Diseases and Hepatic Function",
    18: "Cellular Damage and DNA Repair",
    19: "Sensory Biology and Ion Channels"
}

# ---- Build ONE global topic-color mapping ----
all_topics = set()
for result in topic_results:
    topic_counts = result['topic_counts']
    valid_topics = [topic for topic in topic_counts.index if topic != -1]
    all_topics.update(valid_topics)

all_topics = sorted(all_topics)

if len(all_topics) > len(distinct_colors):
    raise ValueError(
        f"There are {len(all_topics)} topics but only {len(distinct_colors)} colors in Dark24. "
        "You need a larger color palette."
    )

topic_colors = {topic: distinct_colors[i] for i, topic in enumerate(all_topics)}

print("Global topic-color mapping:")
for topic, color in topic_colors.items():
    print(f"Topic {topic}: {color}")

# ---- Process each cluster result ----
for result in topic_results:
    cluster_num = result['cluster']
    topic_counts = result['topic_counts']

    print(f"\nProcessing Cluster {cluster_num}:")
    print(f"Topic counts before filtering:\n{topic_counts}")

    # Filter out topic -1
    if -1 in topic_counts.index:
        filtered_topic_counts = topic_counts.drop(-1).copy()
    else:
        filtered_topic_counts = topic_counts.copy()

    # Skip empty clusters
    if len(filtered_topic_counts) == 0:
        print(f"Skipping cluster {cluster_num} - no valid topics after filtering")
        continue

    print(f"Topic counts after filtering:\n{filtered_topic_counts}")

    # Keep original order for plotting
    topic_index = list(filtered_topic_counts.index)

    # Percentages
    percentages = (filtered_topic_counts / filtered_topic_counts.sum()) * 100

    # Top 3 topics by count
    top3_topics = filtered_topic_counts.sort_values(ascending=False).head(3)

    # Colors using global mapping
    plot_colors = [topic_colors[topic] for topic in topic_index]

    # Create figure with fixed size
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw donut WITHOUT outer labels so pie size stays consistent
    wedges, texts, autotexts = ax.pie(
        filtered_topic_counts.values,
        labels=None,
        autopct=lambda pct: (
            f"{pct:.1f}%\n({int(round(pct / 100 * filtered_topic_counts.sum()))})"
            if pct >= 5 else ""
        ),
        startangle=90,
        colors=plot_colors,
        pctdistance=0.80,
        radius=1.0,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=1),
        textprops={'fontsize': 15}
    )

    # Customize inside percentage labels
    for autotext, percentage in zip(autotexts, percentages):
        if percentage >= 5:
            autotext.set_size(20)
            autotext.set_color("white")
            autotext.set_weight("bold")
        else:
            autotext.set_text("")

    # Add center circle to make it a donut
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    ax.add_artist(centre_circle)

    # ---- Add manual labels for top 3 topics ----
    for topic, value in top3_topics.items():
        summary = topic_summaries.get(topic, "Unknown Topic")
        label = f"Topic {topic}: {summary}"

        # Wrap long label to multiple lines
        label = "\n".join(textwrap.wrap(label, width=32))

        # Locate the corresponding wedge
        wedge_idx = topic_index.index(topic)
        wedge = wedges[wedge_idx]

        # Mid-angle of wedge
        angle = (wedge.theta1 + wedge.theta2) / 2
        angle_rad = np.deg2rad(angle)

        # Start of leader line
        x0 = 1.02 * np.cos(angle_rad)
        y0 = 1.02 * np.sin(angle_rad)

        # End of leader line / text anchor
        x1 = 1.35 * np.cos(angle_rad)
        y1 = 1.35 * np.sin(angle_rad)

        # ---- manual adjustment only for Cluster 15 ----
        if cluster_num == 15:
            if topic == 2:
                y1 += 0.22   # move Topic 2 upward
            if topic == 7:
                y1 -= 0.22   # move Topic 7 downward

        # Draw leader line
        ax.plot([x0, x1], [y0, y1], color='gray', linewidth=1)

        # Align text depending on side
        ha = 'left' if x1 >= 0 else 'right'

        ax.text(
            x1,
            y1,
            label,
            ha=ha,
            va='center',
            fontsize=30,
            fontweight='bold'
        )

    # Keep aspect ratio and fixed limits so donut sizes stay the same
    ax.set_aspect('equal')
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.55, 1.55)

    # Title
    plt.title(
        f'Topic Distribution in Cluster {cluster_num}\n(Total Projects: {filtered_topic_counts.sum()})',
        fontsize=30,
        pad=18
    )

    # Save plot
    output_path = os.path.join(output_dir, f'Cluster_{cluster_num}_Topic_Distribution.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

    print(f"Successfully saved plot for cluster {cluster_num} at {output_path}")

print("\nAll cluster plots generated successfully!")



"""
Dynamic Topic Modeling
"""
# mannual dynamic modeling
topics_full, probabilities_full = topic_model.transform(project_text_full['cleaned_text'], embeddings_array_full)

from collections import Counter
# Count frequency of each topic
topic_counts_full = Counter(topics_full)
# Print sorted counts by topic ID
for topic, count in sorted(topic_counts_full.items()):
    print(f"Topic {topic}: {count}")

project_text_full['topics'] = topics_full
doc_counts_full = project_text_full.groupby('Full Fiscal DateTime').size()


from plotly.colors import qualitative

# Filter out -1 topics
df = project_text_full[project_text_full['topics'] != -1].copy()

# Convert to monthly timestamps
df['YearMonth'] = pd.to_datetime(df['Full Fiscal DateTime']).dt.to_period('M').dt.to_timestamp()

# Count frequency of each topic by month
topic_counts = df.groupby(['YearMonth', 'topics']).size().unstack(fill_value=0)

# Normalize by month
topic_counts_normalized = topic_counts.div(topic_counts.sum(axis=1), axis=0)

# Sort topics by index
topic_counts_normalized = topic_counts_normalized.sort_index(axis=1)

# Use Dark24 color palette (Plotly)
colors = qualitative.Dark24

# Define topic groups
topic_groups = {
    "topics_0_4": list(range(0, 5)),
    "topics_5_9": list(range(5, 10)),
    "topics_10_14": list(range(10, 15)),
    "topics_15_19": list(range(15, 20))
}
for group_name, topic_range in topic_groups.items():
    plt.figure(figsize=(14, 6))

    for i, topic in enumerate(topic_range):
        if topic in topic_counts_normalized.columns:
            color_index = topic  # ensures consistency with Dark24
            plt.plot(topic_counts_normalized.index,
                     topic_counts_normalized[topic],
                     label=f"Topic {topic}",
                     color=colors[color_index],
                     linewidth=2)

    plt.title(f"Normalized Topic Frequencies Over Time", fontsize=20, weight='bold')
    plt.xlabel("Year", fontsize=16)
    plt.ylabel("Normalized Frequency", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.legend(title="Topic", title_fontsize=16, fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save each plot
    plt.savefig(f"time_visualization_{group_name}.png", dpi=300, bbox_inches='tight')


# ================================
# Plot Only Selected Topics
# ================================

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from plotly.colors import qualitative

selected_topics = [0, 1, 2, 6, 12]

# Topic summaries (for better legend)
topic_summaries = {
    0: "Healthcare and Outcomes Research",
    1: "Cellular Biology and Functions",
    2: "Cardiovascular and Cognitive Health",
    6: "Molecular and Cellular Biology",
    12: "Alzheimer’s & Neurodegeneration"
}

# Start from 2009-01-01
topic_counts_normalized_plot = topic_counts_normalized.loc[
    topic_counts_normalized.index >= pd.Timestamp("2009-01-01")
].copy()

# Filter for selected topics
topic_counts_selected = topic_counts_normalized_plot[selected_topics]

# Use Dark24 color palette
colors = qualitative.Dark24

# Define line styles
line_styles = {
    0: 'solid',
    2: 'solid',
    12: 'solid',
    1: 'dashed',
    6: 'dashed'
}

# Create figure
plt.figure(figsize=(14, 6))

# Plot lines
for topic in selected_topics:
    if topic in topic_counts_selected.columns:
        plt.plot(
            topic_counts_selected.index,
            topic_counts_selected[topic],
            label=f"Topic {topic}: {topic_summaries.get(topic, '')}",
            color=colors[topic],
            linewidth=2.5 if line_styles.get(topic) == 'solid' else 2,
            linestyle=line_styles.get(topic, 'solid'),
            alpha=0.9
        )

# Axis formatting
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Add top space for legend
ymax = topic_counts_selected.max().max()
ax.set_ylim(0, ymax * 1.3)

# Labels and title
plt.title("Temporal Trends in Selected Research Topics",
          fontsize=18, weight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Proportion of Projects", fontsize=14)

# Grid
plt.grid(True, alpha=0.2)

# Tick styling
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# ================================
# Legend
# ================================
plt.legend(
    title="Topic",
    title_fontsize=13,
    fontsize=11,
    loc='upper left',
    bbox_to_anchor=(0.10, 0.98),
    frameon=True,
    facecolor='white',
    edgecolor='gray',
    framealpha=0.9
)

# Layout
plt.tight_layout()

# Save figure
plt.savefig(
    "time_visualization_selected_topics.png",
    dpi=300,
    bbox_inches='tight'
)