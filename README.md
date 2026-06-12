# NIH-MPINet

This repository contains code and data for constructing the NIH multiple principal investigator (multi-PI) collaboration network and performing community detection and topic modeling.

## Data

The processed grant-level dataset, PI-level dataset, and the corresponding network objects are available at:
https://huggingface.co/datasets/LiLabUNC/NIH-MPINet

## Code description

### read_data.py
Cleans NIH RePORTER data and prepares the grant-level dataset.

### covariate_department_stop.py
Scrapes PI affiliation information from PubMed.

### graph.py
Constructs the PI collaboration network and identifies the largest connected component.

### summary_statistics.py
Generates summary statistics for the grant-level and PI-level datasets.

### community_detection.py
Detects network communities using the Leiden algorithm.

### topic_modeling_nih_Bertopic.py
Performs topic modeling on grant titles and abstracts using BERTopic with BioSentVec embeddings.

