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
topic_visualization.write_image("/Users/shicuiran/PycharmProjects/NIH_network/topic_model/topic_visualization.png")
barchart_visualization.write_image("/Users/shicuiran/PycharmProjects/NIH_network/topic_model/barchart_visualization.png")
heatmap.write_image("/Users/shicuiran/PycharmProjects/NIH_network/topic_model/heatmap.png")

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
barchart_visualization.write_image("/Users/shicuiran/PycharmProjects/NIH_network/topic_model/barchart_visualization_full1.png")



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
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from itertools import cycle

# Replace this with your actual data retrieval line
topic_counts4 = topic_results[4]['topic_counts']  # both cluster and topic start from 0

# Filter out the '-1' topic if it exists
if -1 in topic_counts4:
    filtered_topic_counts = topic_counts4.drop(-1)
else:
    filtered_topic_counts = topic_counts4

# Use Plotly's Dark24 qualitative color palette
distinct_colors = px.colors.qualitative.Dark24
color_cycle = cycle(distinct_colors)  # Create a cycling iterator over the colors

# Map each unique topic to a color
topic_colors = {topic: next(color_cycle) for topic in sorted(set(filtered_topic_counts.index))}

# Use the topic_colors to assign colors to the topics in filtered_topic_counts
plot_colors = [topic_colors[topic] for topic in filtered_topic_counts.index]

# Create a donut plot
fig, ax = plt.subplots(figsize=(10, 8))  # Increase the figure size
wedges, texts, autotexts = ax.pie(
    filtered_topic_counts,
    labels=filtered_topic_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=plot_colors,  # Use mapped colors
    pctdistance=0.85,
    wedgeprops=dict(width=0.3)
)

# Customize text size and color
plt.setp(texts, size=8, weight="bold")
plt.setp(autotexts, size=8, color="white")

# Draw a circle at the center of the pie to make it look like a donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')

plt.title('Topic Distribution in Cluster 4', fontsize=14)

# Save the plot as a PNG file
plt.savefig('/Users/shicuiran/PycharmProjects/NIH_network/topic_model/Topic_Distribution_Donut_Plot_cluster4.png',
            bbox_inches='tight',
            dpi=300)  # Added higher DPI for better quality




# Write donut plots in a loop
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from itertools import cycle
import os
import numpy as np

# Create output directory if it doesn't exist
output_dir = '/Users/shicuiran/PycharmProjects/NIH_network/topic_model/cluster_plots'
os.makedirs(output_dir, exist_ok=True)

# Use Plotly's Dark24 qualitative color palette
distinct_colors = px.colors.qualitative.Dark24

# Process each cluster result
for result in topic_results:
    cluster_num = result['cluster']
    topic_counts = result['topic_counts']

    print(f"\nProcessing Cluster {cluster_num}:")
    print(f"Topic counts before filtering:\n{topic_counts}")

    # Filter out the '-1' topic if it exists
    if -1 in topic_counts.index:
        filtered_topic_counts = topic_counts.drop(-1)
    else:
        filtered_topic_counts = topic_counts.copy()

    # Skip clusters with no valid topics
    if len(filtered_topic_counts) == 0:
        print(f"Skipping cluster {cluster_num} - no valid topics after filtering")
        continue

    print(f"Topic counts after filtering:\n{filtered_topic_counts}")

    # Calculate percentages
    percentages = (filtered_topic_counts / filtered_topic_counts.sum()) * 100

    # Create color cycle and map topics to colors
    color_cycle = cycle(distinct_colors)
    topic_colors = {topic: next(color_cycle) for topic in sorted(filtered_topic_counts.index)}
    plot_colors = [topic_colors[topic] for topic in filtered_topic_counts.index]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create donut plot - hide labels for small percentages (<5%)
    wedges, texts, autotexts = ax.pie(
        filtered_topic_counts,
        labels=[f"Topic {topic}" if percentages[topic] >= 5 else "" for topic in filtered_topic_counts.index],
        autopct=lambda pct: f"{pct:.1f}%\n({int(pct / 100 * filtered_topic_counts.sum())})" if pct >= 5 else "",
        startangle=90,
        colors=plot_colors,
        pctdistance=0.80,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=1),
        textprops={'fontsize': 9}
    )

    # Customize text - only for visible labels
    for text, percentage in zip(texts, percentages):
        if percentage >= 5:
            text.set_size(16)
            text.set_weight("bold")
        else:
            text.set_text("")  # Ensure empty labels are truly empty

    for autotext, percentage in zip(autotexts, percentages):
        if percentage >= 5:
            autotext.set_size(14)
            autotext.set_color("white")
            autotext.set_weight("bold")
        else:
            autotext.set_text("")  # Hide small percentage labels

    # Add center circle to make it a donut
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    ax.add_artist(centre_circle)
    ax.axis('equal')

    # Add title and adjust layout
    plt.title(f'Topic Distribution in Cluster {cluster_num}\n(Total Projects: {filtered_topic_counts.sum()})',
              fontsize=20, pad=20)
    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, f'Cluster_{cluster_num}_Topic_Distribution.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()  # Close the figure to free memory

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

import pandas as pd
import matplotlib.pyplot as plt
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
selected_topics = [0, 2, 6, 12]

# Filter for selected topics
topic_counts_selected = topic_counts_normalized[selected_topics]

# Use Dark24 color palette
colors = qualitative.Dark24

# Plot
plt.figure(figsize=(14, 6))
for topic in selected_topics:
    if topic in topic_counts_selected.columns:
        plt.plot(topic_counts_selected.index,
                 topic_counts_selected[topic],
                 label=f"Topic {topic}",
                 color=colors[topic],
                 linewidth=2)

plt.title("Normalized Frequencies for Selected Topics (0, 1, 2, 6, 12)", fontsize=20, weight='bold')
plt.xlabel("Year", fontsize=16)
plt.ylabel("Normalized Frequency", fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.legend(title="Topic", title_fontsize=16, fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the figure
plt.savefig("time_visualization_selected_topics.png", dpi=300, bbox_inches='tight')
