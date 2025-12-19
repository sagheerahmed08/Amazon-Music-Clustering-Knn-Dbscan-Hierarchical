"""
Cluster Evaluation Module for Amazon Music Clustering Project.

This module provides:
- Cluster quality metrics (Silhouette Score, Davies-Bouldin Index)
- Cluster interpretation based on feature averages
- Visualization of cluster characteristics
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, List, Optional, Tuple


def evaluate_clusters(df: pd.DataFrame, features: List[str], 
                     cluster_column: str = "Cluster_KMeans") -> Dict[str, float]:
    """
    Evaluate cluster quality using multiple metrics.
    
    Args:
        df: DataFrame with features and cluster labels
        features: List of feature names used for clustering
        cluster_column: Name of the column containing cluster labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\n🔹 Evaluating {cluster_column} clustering quality...")
    
    # Filter features to only those that exist in the dataframe
    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        raise ValueError(f"No valid features found. Available columns: {df.columns.tolist()}")
    
    if len(valid_features) < len(features):
        missing = set(features) - set(valid_features)
        print(f"⚠️ Warning: Some features not found: {missing}")
    
    # Extract feature matrix and labels
    X = df[valid_features].values
    labels = df[cluster_column].values
    
    # Check for valid clusters
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    if n_clusters < 2:
        print("⚠️ Warning: Less than 2 clusters found. Metrics may be unreliable.")
        return {
            'silhouette_score': 0.0,
            'davies_bouldin_index': float('inf'),
            'n_clusters': n_clusters
        }
    
    # Compute metrics
    silhouette = silhouette_score(X, labels)
    db_index = davies_bouldin_score(X, labels)
    
    metrics = {
        'silhouette_score': silhouette,
        'davies_bouldin_index': db_index,
        'n_clusters': n_clusters
    }
    
    print(f"✅ Silhouette Score: {silhouette:.4f}")
    print(f"   (Higher is better, range: -1 to 1)")
    print(f"✅ Davies-Bouldin Index: {db_index:.4f}")
    print(f"   (Lower is better)")
    print(f"✅ Number of clusters: {n_clusters}")
    
    return metrics


def interpret_clusters(df: pd.DataFrame, features: List[str],
                      cluster_column: str = "Cluster_KMeans") -> pd.DataFrame:
    """
    Calculate mean feature values for each cluster to interpret characteristics.
    
    Args:
        df: DataFrame with features and cluster labels
        features: List of feature names
        cluster_column: Name of the column containing cluster labels
        
    Returns:
        DataFrame with cluster-wise feature averages
    """
    print(f"\n🔹 Computing cluster-wise feature averages...")
    
    cluster_summary = df.groupby(cluster_column)[features].mean()
    
    print("\nCluster Summary (Mean Feature Values):")
    print(cluster_summary.round(3))
    
    return cluster_summary


def visualize_cluster_profiles(cluster_summary: pd.DataFrame,
                               save_path: Optional[str] = None) -> None:
    """
    Create heatmap visualization of cluster profiles.
    
    Args:
        cluster_summary: DataFrame with cluster-wise feature averages
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        cluster_summary, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Average Feature Values per Cluster", fontsize=14, pad=12)
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Cluster ID", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Cluster profile heatmap saved to: {save_path}")
    
    plt.show()


def visualize_cluster_sizes(df: pd.DataFrame, 
                           cluster_column: str = "Cluster_KMeans",
                           save_path: Optional[str] = None) -> None:
    """
    Visualize the distribution of cluster sizes.
    
    Args:
        df: DataFrame with cluster labels
        cluster_column: Name of the column containing cluster labels
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(8, 5))
    cluster_counts = df[cluster_column].value_counts().sort_index()
    
    bars = plt.bar(cluster_counts.index.astype(str), cluster_counts.values, 
                   color=plt.cm.Set2(range(len(cluster_counts))))
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.title(f"Cluster Size Distribution ({cluster_column})", fontsize=14, pad=12)
    plt.xlabel("Cluster ID", fontsize=12)
    plt.ylabel("Number of Songs", fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Cluster size distribution saved to: {save_path}")
    
    plt.show()


def generate_cluster_interpretations(cluster_summary: pd.DataFrame,
                                     features: List[str]) -> Dict[int, Dict[str, List[str]]]:
    """
    Generate human-readable interpretations for each cluster.
    
    Args:
        cluster_summary: DataFrame with cluster-wise feature averages
        features: List of feature names
        
    Returns:
        Dictionary mapping cluster ID to interpretation details
    """
    interpretations = {}
    
    for cluster_id, row in cluster_summary.iterrows():
        # Calculate mean across all features for this cluster
        cluster_mean = row.mean()
        
        # Features above cluster mean
        high_features = row[row > cluster_mean].index.tolist()
        # Features below cluster mean
        low_features = row[row < cluster_mean].index.tolist()
        
        # Generate interpretation
        interpretation = []
        if 'energy' in high_features and 'danceability' in high_features:
            interpretation.append("Upbeat / Party tracks")
        if 'acousticness' in high_features:
            interpretation.append("Chill or Acoustic songs")
        if 'instrumentalness' in high_features:
            interpretation.append("Instrumental / Ambient tracks")
        if 'valence' in high_features:
            interpretation.append("Positive mood")
        if 'speechiness' in high_features:
            interpretation.append("Speech-heavy / Rap")
        if not interpretation:
            interpretation.append("Mixed characteristics")
        
        interpretations[cluster_id] = {
            'high_features': high_features,
            'low_features': low_features,
            'interpretation': interpretation
        }
    
    return interpretations


def print_cluster_interpretations(interpretations: Dict[int, Dict[str, List[str]]]) -> None:
    """
    Print cluster interpretations in a readable format.
    
    Args:
        interpretations: Dictionary from generate_cluster_interpretations
    """
    print("\n🎵 Cluster Interpretation:")
    print("=" * 50)
    
    for cluster_id in sorted(interpretations.keys()):
        info = interpretations[cluster_id]
        print(f"\n🎧 Cluster {cluster_id}:")
        print(f"   🔺 High in: {', '.join(info['high_features'][:5])}")
        if len(info['high_features']) > 5:
            print(f"      ... and {len(info['high_features']) - 5} more")
        print(f"   🔻 Low in: {', '.join(info['low_features'][:5])}")
        if len(info['low_features']) > 5:
            print(f"      ... and {len(info['low_features']) - 5} more")
        print(f"   💡 Possible interpretation: {', '.join(info['interpretation'])}")


def evaluation_pipeline(clustered_data_path: str,
                       features: List[str],
                       cluster_column: str = "Cluster_KMeans",
                       output_dir: str = "Data/Processed") -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Complete evaluation pipeline: metrics, interpretation, visualization.
    
    Args:
        clustered_data_path: Path to clustered dataset CSV
        features: List of feature names
        cluster_column: Name of cluster column to evaluate
        output_dir: Directory to save outputs
        
    Returns:
        Tuple of (metrics dictionary, cluster summary DataFrame)
    """
    # Load clustered data
    print("\n🔹 Loading clustered dataset...")
    if not os.path.exists(clustered_data_path):
        raise FileNotFoundError(f"Clustered data file not found: {clustered_data_path}")
    
    df = pd.read_csv(clustered_data_path)
    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    
    # Evaluate clusters
    metrics = evaluate_clusters(df, features, cluster_column)
    
    # Interpret clusters
    cluster_summary = interpret_clusters(df, features, cluster_column)
    
    # Save cluster summary
    summary_path = os.path.join(output_dir, "cluster_summary.csv")
    cluster_summary.to_csv(summary_path, index=True)
    print(f"\n✅ Cluster summary saved to: {summary_path}")
    
    # Visualizations
    print("\n🔹 Creating visualizations...")
    visualize_cluster_profiles(
        cluster_summary,
        save_path=os.path.join(output_dir, "cluster_profiles_heatmap.png")
    )
    
    visualize_cluster_sizes(
        df,
        cluster_column=cluster_column,
        save_path=os.path.join(output_dir, "cluster_size_distribution.png")
    )
    
    # Generate and print interpretations
    interpretations = generate_cluster_interpretations(cluster_summary, features)
    print_cluster_interpretations(interpretations)
    
    print("\n✅ Cluster Evaluation & Interpretation Completed Successfully!")
    
    return metrics, cluster_summary


if __name__ == "__main__":
    # Main execution
    clustered_data_path = "Data/Processed/clustered_songs.csv"
    
    # Load data to get actual features (after transformations)
    df_temp = pd.read_csv(clustered_data_path)
    features = [col for col in df_temp.columns 
                if not col.startswith('Cluster_') and col not in ['PCA1', 'PCA2', 'TSNE1', 'TSNE2']]
    
    try:
        metrics, summary = evaluation_pipeline(
            clustered_data_path,
            features,
            cluster_column="Cluster_KMeans"
        )
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise
