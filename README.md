# Amazon Music Clustering

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47.1-orange)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-yellow)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red)](https://xgboost.readthedocs.io/)
[![AWS S3](https://img.shields.io/badge/AWS-S3-brightgreen)](https://aws.amazon.com/s3/)

## Project Overview

This repository analyzes audio-track features to discover natural groups of songs using unsupervised learning. The pipeline cleans and transforms raw track data, scales features, reduces dimensionality (PCA/t-SNE) and applies clustering algorithms (KNN-based analyses for neighbor inspection, K-Means, DBSCAN and hierarchical clustering). Outputs include labeled datasets, visualizations, and a written summary.

## Quick Links

- Notebook: `src/Notebook.ipynb`
- Preprocessing: `src/preprocessing.py`
- Clustering: `src/clustering.py`
- Evaluation: `src/evaluation.py`
- Visualization helpers: `src/visualization.py`

## Folder Structure

```
.
├── Data/
│   ├── Raw/
│   │   └── single_genre_artists.csv
│   └── Processed/
│       ├── transformed_features.csv
│       ├── scaled_features.csv
│       ├── final_clustered_data.csv
│       ├── summary_report.txt
│       └── *.png (visualizations)
├── src/
│   ├── Notebook.ipynb
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── main.py
├── requirements.txt
├── CODE_IMPROVEMENTS.md
└── README.md
```

## Features (columns) — explanation

Numerical features

- `duration_ms` — Song length in milliseconds. Long tracks may correlate with different structures (e.g., instrumental, live).
- `danceability` — How suitable a track is for dancing (0.0–1.0). Higher = more danceable.
- `energy` — Perceptual intensity and activity (0.0–1.0). High energy often means louder and faster.
- `loudness` — Average loudness in dB. Negative values; higher (closer to 0) means louder.
- `speechiness` — Detects presence of spoken words; high values indicate talk or rap.
- `acousticness` — Likelihood the track is acoustic (0.0–1.0).
- `instrumentalness` — Likelihood track has no vocals (0.0–1.0). Good for ambient/instrumental detection.
- `liveness` — Probability the track was performed live (0.0–1.0).
- `valence` — Musical positiveness (0.0–1.0). High valence = more upbeat/happy.
- `tempo` — Beats per minute (BPM). Affects perceived energy and genre.

Categorical features

- `key` — Musical key (0–11). Useful for tonal grouping.
- `mode` — Major (1) or minor (0) tonal quality.
- `time_signature` — Beats per bar (commonly 3/4, 4/4 etc.).

Why these features?

These features are standard audio descriptors (from Spotify/API-like features) capturing rhythm, timbre, harmony, and perceptual qualities. Combining them helps separate songs by mood, instrumentation, and production style.

## Selected Features for Clustering

- Core numerical features used by the pipeline: `duration_ms`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`.
- Categorical features (`key`, `mode`, `time_signature`) are one-hot encoded and appended to the feature set.

Selection rationale

- `danceability`, `energy`, `valence`, and `tempo` are strong indicators of perceived mood and activity — useful for separating upbeat vs chill clusters.
- `acousticness` and `instrumentalness` help identify acoustic/instrumental clusters.
- `speechiness` isolates spoken-word/heavy vocal tracks.

## Dimensionality Reduction

PCA
- Purpose: Reduce feature dimensionality while preserving variance. PCA is used to speed clustering and provide interpretable principal components. Typical pipeline: standardize features → PCA (6–8 components) → clustering on PCA space.

t-SNE
- Purpose: Nonlinear projection for 2D visualization of cluster structure. t-SNE is applied to PCA output (recommended) for stability and speed.

Usage notes
- Set `random_state` for reproducibility. Use explained variance from PCA to choose number of components.

## Clustering Methods Explained

K-Nearest Neighbors (KNN)
- Not an unsupervised clustering method but useful for neighborhood inspection and recommendations (e.g., find nearest songs to a seed track). KNN on scaled features can support labeling or nearest-neighbor retrieval.

MiniBatch K-Means
- Fast variant of K-Means suited for larger datasets. Use elbow and silhouette metrics to choose `k`. Produces compact, spherical clusters.

DBSCAN
- Density-based clustering that detects arbitrarily-shaped clusters and marks noise points. Key hyperparameters: `eps` (neighborhood radius) and `min_samples`. Good for detecting outlier tracks that don't belong to dense clusters.

Hierarchical (Agglomerative)
- Builds a merge tree of clusters; useful to inspect multi-scale structure via a dendrogram. Can cut the tree at different levels to obtain different numbers of clusters.

How methods are used in this repo
- Run experiments with K-Means across a range of `k` values, inspect elbow and silhouette scores.
- Use DBSCAN to identify dense clusters and noise (requires tuning `eps`).
- Generate dendrograms from hierarchical linkage for exploratory analysis and to choose hierarchical groupings.

## Evaluation Metrics

- Silhouette score — cohesion vs separation of clusters (range -1..1). Higher is better.
- Davies–Bouldin index — average similarity measure of each cluster with its most similar cluster. Lower is better.

## Outputs

- `Data/Processed/transformed_features.csv` — after log transforms and one-hot encoding
- `Data/Processed/scaled_features.csv` — standardized numerical features
- `Data/Processed/final_clustered_data.csv` — original data merged with cluster labels
- Visualizations: `pca_visualization.png`, `tsne_visualization.png`, `elbow_method.png`, `silhouette_scores.png`, `dendrogram.png`, `cluster_profile_heatmap.png`
- `Data/Processed/summary_report.txt` — human-readable summary of clusters

## How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Open `src/Notebook.ipynb` and run cells sequentially for an exploratory workflow, or run `src/main.py` for the scripted pipeline.

## Recommendations & Tips

- Preprocess: remove duplicates and invalid rows before fitting models.
- For PCA: choose number of components by explained variance (e.g., keep components that cover 80–95% variance).
- For DBSCAN: create a k-distance plot to select `eps`.
- For visualizations: use a sample for dendrograms to keep plots readable.

## Contact / Next steps

If you want, I can:

- Add badges for CI and package versions.
- Write `requirements.txt` into the repo and pin versions.
- Commit and push the project to your GitHub remote (you'll need to provide authentication or run the `git push` locally).

---

Created for the Amazon Music Clustering project — let me know if you want shorter or longer README variants.
