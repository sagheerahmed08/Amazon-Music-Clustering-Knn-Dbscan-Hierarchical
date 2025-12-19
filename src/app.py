"""
Streamlit Dashboard for Amazon Music Clustering Project.

This interactive dashboard provides:
- Data exploration and preprocessing insights
- Cluster evaluation metrics
- Visualizations of clustering results
- Final results and downloadable reports
"""

import os
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


# Configuration
DATA_DIR = "Data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "Raw", "single_genre_artists.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "Processed")

# Features will be dynamically extracted from data
# to handle transformations (log transform, one-hot encoding)
def get_features_from_data(df: pd.DataFrame) -> list:
    """Extract feature columns from DataFrame, excluding metadata columns."""
    exclude_cols = ['id_songs', 'name_song', 'id_artists', 'release_date', 
                   'genres', 'name_artists', 'Cluster_KMeans', 'Cluster_DBSCAN',
                   'Cluster_Hierarchical', 'PCA1', 'PCA2', 'TSNE1', 'TSNE2']
    return [col for col in df.columns if col not in exclude_cols]


@st.cache_data
def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load and cache data from CSV file."""
    try:
        if not os.path.exists(file_path):
            return None
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def check_file_exists(file_path: str) -> bool:
    """Check if a file exists and show error if not."""
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        st.info("Please run the preprocessing and clustering pipelines first.")
        return False
    return True


# --------------------------
# PAGE CONFIGURATION
# --------------------------
st.set_page_config(
    page_title="Amazon Music Clustering",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎵 Amazon Music Clustering Dashboard")

# --------------------------
# SIDEBAR NAVIGATION
# --------------------------
with st.sidebar:
    select = option_menu(
        "Main Menu",
        ["Home", "Data Exploration", "Cluster Evaluation", "Visualizations", "Final Results"],
        icons=["house", "bar-chart", "activity", "pie-chart", "list-task"],
        menu_icon="music-note",
        default_index=0
    )

# --------------------------
# HOME PAGE
# --------------------------
if select == "Home":
    st.header("Welcome to the Amazon Music Clustering Project 🎧")
    st.markdown("""
    This interactive dashboard visualizes clustering analysis on Amazon Music dataset.
    
    **Project Highlights:**
    - ✅ Data preprocessing, normalization, and feature selection
    - ✅ Dimensionality reduction using PCA and t-SNE
    - ✅ Clustering with MiniBatch K-Means, DBSCAN, and Hierarchical models
    - ✅ Cluster evaluation using Silhouette & Davies-Bouldin metrics
    - ✅ Visualization of clusters and summary interpretation
    
    **Navigation:**
    - **Data Exploration**: Explore the raw dataset and feature distributions
    - **Cluster Evaluation**: View cluster quality metrics and interpretations
    - **Visualizations**: See 2D projections and cluster comparisons
    - **Final Results**: Download final datasets and reports
    
    Use the **sidebar** to navigate through sections.
    """)
    
    # Quick stats
    if check_file_exists(RAW_DATA_PATH):
        df = load_data(RAW_DATA_PATH)
        if df is not None:
            st.success(f"✅ Dataset loaded: {len(df):,} songs")

# --------------------------
# DATA EXPLORATION
# --------------------------
elif select == "Data Exploration":
    st.header("🔍 Data Exploration & Preprocessing")
    
    if not check_file_exists(RAW_DATA_PATH):
        st.stop()
    
    df = load_data(RAW_DATA_PATH)
    if df is None:
        st.stop()
    
    # Data Overview Metrics
    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    missing_value = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
        st.metric("Numeric Features", df.select_dtypes(include=['int64', 'float64']).shape[1])
    
    with col2:
        st.metric("Total Columns", df.shape[1])
        st.metric("Categorical Features", df.select_dtypes(include=['object']).shape[1])
    
    with col3:
        st.metric("Missing Values", f"{missing_value:,}")
        st.metric("Duplicate Rows", f"{duplicates:,}")
    
    # Raw Dataset Preview
    with st.expander("📄 Raw Dataset Preview", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Showing first 100 rows of {len(df):,} total rows")
    
    # Statistical Summary
    with st.expander("📊 Statistical Summary", expanded=False):
        st.dataframe(df.describe(), use_container_width=True)
    
    # Feature Selection
    st.subheader("🎯 Selected Features for Clustering")
    
    # Dynamically extract features from data
    available_features = get_features_from_data(df)
    numerical_column = [col for col in available_features if col in [
        'duration_ms', 'danceability', 'energy',
        'loudness', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo','popularity_songs','explicit','popularity_artists','followers'
    ]]
    features = [
        'duration_ms', 'danceability', 'energy',
        'loudness', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo'
    ]
    if available_features and features:
        available_features_df = pd.DataFrame({
                "S.No": range(1, len(available_features) + 1),
                "Feature Name": available_features,
                "Data Type": [str(df[col].dtype) for col in available_features],
                "Unique Values": [df[col].nunique() for col in available_features],
                "Count": [df[col].shape[0] for col in available_features],
                "Numerical_column": [col in numerical_column for col in available_features],
                "Categorical_column": [col not in numerical_column for col in available_features]
            })
        st.dataframe(available_features_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No features found in dataset")
    
    # Feature Distributions
    st.subheader("📊 Feature Distributions")
    
    if len(available_features) > 0:
        # Create 3 columns for visualization
        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        col1, col2 = st.columns(2)
        with col1:
            plt.figure(figsize=(15, 5 * n_rows))
            for i, col in enumerate(available_features, 1):
                plt.subplot(n_rows, n_cols, i)
                sns.histplot(df[col], kde=True, bins=30, color='skyblue')
                plt.title(col.capitalize(), fontsize=12)
                plt.xlabel(col)
                plt.ylabel('Frequency')
        
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)
        with col2:
            st.pyplot(plt)
        col1, col2, col3 = st.columns(3)
        
        n_features = len(available_features)
        features_per_col = (n_features + 2) // 3
    
        with col1:
            for i in range(0, min(features_per_col, n_features)):
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.histplot(df[available_features[i]], kde=True, bins=30, 
                           color='skyblue', ax=ax)
                ax.set_title(available_features[i].capitalize(), fontsize=10)
                ax.set_xlabel("")
                ax.set_ylabel("")
                st.pyplot(fig, use_container_width=True)
        
        with col2:
            start_idx = features_per_col
            end_idx = min(2 * features_per_col, n_features)
            for i in range(start_idx, end_idx):
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.histplot(df[available_features[i]], kde=True, bins=30, 
                           color='lightgreen', ax=ax)
                ax.set_title(available_features[i].capitalize(), fontsize=10)
                ax.set_xlabel("")
                ax.set_ylabel("")
                st.pyplot(fig, use_container_width=True)
        
        with col3:
            start_idx = 2 * features_per_col
            for i in range(start_idx, n_features):
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.histplot(df[available_features[i]], kde=True, bins=30, 
                           color='lightcoral', ax=ax)
                ax.set_title(available_features[i].capitalize(), fontsize=10)
                ax.set_xlabel("")
                ax.set_ylabel("")
                st.pyplot(fig, use_container_width=True)
        
        # Correlation Heatmap
        st.subheader("🔥 Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df[features].corr(),
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            center=0,
            square=True,
            ax=ax
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=14, pad=12)
        st.pyplot(fig)
        
        st.subheader("🔥 Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df[features].corr(),
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            center=0,
            square=True,
            ax=ax
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=14, pad=12)
        st.pyplot(fig)
        # PCA and t-SNE Visualizations
        st.subheader("📉 Dimensionality Reduction Visualizations")
        col1, col2 = st.columns(2)
        
        pca_path = "Data/Processed/pca_visualization.png"
        tsne_path = "Data/Processed/tsne_visualization.png"
        
        with col1:
            if os.path.exists(pca_path):
                img1 = Image.open(pca_path)
                st.image(img1, caption="PCA Visualization (2 Components)", use_container_width=True)
            else:
                st.info("PCA visualization not found. Run preprocessing pipeline to generate.")
        
        with col2:
            if os.path.exists(tsne_path):
                img2 = Image.open(tsne_path)
                st.image(img2, caption="t-SNE Visualization", use_container_width=True)
            else:
                st.info("t-SNE visualization not found. Run preprocessing pipeline to generate.")

# --------------------------
# CLUSTER EVALUATION
# --------------------------
elif select == "Cluster Evaluation":
    st.header("📊 Cluster Evaluation & Interpretation")
    
    scaled_data_path = os.path.join(PROCESSED_DIR, "scaled_features.csv")
    cluster_summary_path = os.path.join(PROCESSED_DIR, "cluster_summary.csv")
    clustered_data_path = os.path.join(PROCESSED_DIR, "clustered_songs.csv")
    
    # Load scaled data
    if check_file_exists(scaled_data_path):
        scaled_data = load_data(scaled_data_path)
        if scaled_data is not None:
            with st.expander("📄 Scaled Features Preview", expanded=False):
                st.dataframe(scaled_data.head(100), use_container_width=True)
    
    # Cluster Summary
    if check_file_exists(cluster_summary_path):
        cluster_summary = load_data(cluster_summary_path)
        if cluster_summary is not None:
            st.subheader("🔹 Cluster-wise Feature Averages")
            st.dataframe(cluster_summary, use_container_width=True)
            
            # Heatmap
            st.subheader("🔥 Heatmap of Average Features per Cluster")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Handle index column
            if 'Cluster_KMeans' in cluster_summary.columns:
                cluster_summary_viz = cluster_summary.set_index("Cluster_KMeans")
            else:
                cluster_summary_viz = cluster_summary
            
            sns.heatmap(
                cluster_summary_viz,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                center=0,
                linewidths=0.5,
                ax=ax
            )
            ax.set_title("Average Feature Values per Cluster", fontsize=14, pad=12)
            st.pyplot(fig)
    
    # Cluster Size Distribution
    if check_file_exists(clustered_data_path):
        df_clustered = load_data(clustered_data_path)
        if df_clustered is not None and 'Cluster_KMeans' in df_clustered.columns:
            st.subheader("📈 Cluster Size Distribution")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            cluster_counts = df_clustered['Cluster_KMeans'].value_counts().sort_index()
            
            bars = ax2.bar(cluster_counts.index.astype(str), cluster_counts.values,
                          color=plt.cm.Set2(range(len(cluster_counts))))
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10)
            
            ax2.set_title("Cluster Size Distribution (K-Means)", fontsize=14, pad=12)
            ax2.set_xlabel("Cluster ID", fontsize=12)
            ax2.set_ylabel("Number of Songs", fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig2)

# --------------------------
# VISUALIZATIONS
# --------------------------
elif select == "Visualizations":
    st.header("🎨 Visualizing Clusters")
    
    pca_path = "Data/Processed/pca_clusters.png"
    tsne_path = "Data/Processed/tsne_clusters.png"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 PCA-based 2D Cluster Visualization")
        if os.path.exists(pca_path):
            img1 = Image.open(pca_path)
            st.image(img1, caption="PCA 2D Visualization of Clusters", use_container_width=True)
        else:
            st.info("PCA visualization not found. Run visualization pipeline to generate.")
    
    with col2:
        st.subheader("📉 t-SNE-based 2D Cluster Visualization")
        if os.path.exists(tsne_path):
            img2 = Image.open(tsne_path)
            st.image(img2, caption="t-SNE 2D Visualization of Clusters", use_container_width=True)
        else:
            st.info("t-SNE visualization not found. Run visualization pipeline to generate.")
    
    st.markdown("""
    These visualizations represent how songs are grouped into clusters in reduced dimensions.
    - **PCA**: Linear dimensionality reduction preserving maximum variance
    - **t-SNE**: Non-linear dimensionality reduction preserving local neighborhood structure
    """)

# --------------------------
# FINAL RESULTS & SUMMARY
# --------------------------
elif select == "Final Results":
    st.header("📜 Final Results & Cluster Insights")
    
    final_data_path = os.path.join(PROCESSED_DIR, "final_clustered_dataset.csv")
    top_tracks_path = os.path.join(PROCESSED_DIR, "top_tracks_per_cluster.csv")
    report_path = os.path.join(PROCESSED_DIR, "final_cluster_report.txt")
    
    # Top Tracks
    if check_file_exists(top_tracks_path):
        top_tracks = load_data(top_tracks_path)
        if top_tracks is not None:
            st.subheader("🎧 Top 5 Songs per Cluster")
            st.dataframe(top_tracks, use_container_width=True, hide_index=True)
    
    # Summary Report
    if check_file_exists(report_path):
        st.subheader("📘 Cluster Summary Report")
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = f.read()
            st.text_area("Summary Report", report, height=500, label_visibility="collapsed")
        except Exception as e:
            st.error(f"Error reading report: {e}")
    else:
        st.info("Summary report not found. Run final analysis pipeline to generate.")
    
    # Download Buttons
    if check_file_exists(final_data_path):
        final_df = load_data(final_data_path)
        if final_df is not None:
            st.subheader("📥 Downloads")
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="⬇️ Download Final Clustered Dataset (CSV)",
                    data=final_df.to_csv(index=False).encode("utf-8"),
                    file_name="final_clustered_dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                if os.path.exists(report_path):
                    try:
                        with open(report_path, "r", encoding="utf-8") as f:
                            report = f.read()
                        st.download_button(
                            label="⬇️ Download Summary Report (TXT)",
                            data=report.encode("utf-8"),
                            file_name="final_cluster_report.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    except Exception:
                        pass
    
    st.success("✅ All available reports and visualizations loaded successfully!")
