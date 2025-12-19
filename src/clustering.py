"""
Clustering Module for Amazon Music Clustering Project.

This module implements multiple clustering algorithms:
- K-Means (MiniBatch for efficiency)
- DBSCAN
- Hierarchical Clustering
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Tuple, List, Optional


def find_optimal_k_elbow_silhouette(df_scaled: pd.DataFrame, 
                                     k_range: range = range(2, 11),
                                     save_plots: bool = True,
                                     output_dir: str = "Data/Processed") -> int:
    """
    Find optimal number of clusters using Elbow Method and Silhouette Score.
    
    Args:
        df_scaled: Scaled feature DataFrame
        k_range: Range of k values to test
        save_plots: Whether to save plots
        output_dir: Directory to save plots
        
    Returns:
        Optimal k value based on silhouette score
    """
    print(f"\n🔹 Evaluating optimal K using Elbow & Silhouette methods...")
    print(f"   Testing k values: {list(k_range)}")
    
    inertia = []
    sil_scores = []
    
    for k in k_range:
        kmeans = MiniBatchKMeans(
            n_clusters=k, 
            random_state=42, 
            batch_size=4096, 
            max_iter=200,
            n_init=10
        )
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(df_scaled, kmeans.labels_))
        print(f"   k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_scores[-1]:.4f}")
    
    # Elbow Method Plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Inertia (SSE)", fontsize=12)
    plt.title("Elbow Method for Optimal k", fontsize=14, pad=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plots:
        elbow_path = os.path.join(output_dir, "elbow_method.png")
        plt.savefig(elbow_path, dpi=300, bbox_inches='tight')
        print(f"✅ Elbow plot saved to: {elbow_path}")
    
    plt.show()
    
    # Silhouette Scores Plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, sil_scores, 'ro-', linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Silhouette Score", fontsize=12)
    plt.title("Silhouette Score vs Number of Clusters", fontsize=14, pad=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plots:
        sil_path = os.path.join(output_dir, "silhouette_scores.png")
        plt.savefig(sil_path, dpi=300, bbox_inches='tight')
        print(f"✅ Silhouette plot saved to: {sil_path}")
    
    plt.show()
    
    # Find best k (highest silhouette score)
    best_k_idx = np.argmax(sil_scores)
    best_k = list(k_range)[best_k_idx]
    
    print(f"\n✅ Optimal number of clusters (k): {best_k}")
    print(f"   Silhouette Score: {sil_scores[best_k_idx]:.4f}")
    
    return best_k


def apply_kmeans(df_scaled: pd.DataFrame, n_clusters: int,
                random_state: int = 42) -> Tuple[pd.DataFrame, MiniBatchKMeans]:
    """
    Apply K-Means clustering to the scaled data.
    
    Args:
        df_scaled: Scaled feature DataFrame
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (DataFrame with cluster labels, fitted KMeans model)
    """
    print(f"\n🔹 Running MiniBatchKMeans with k={n_clusters}...")
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=4096,
        max_iter=300,
        n_init=10
    )
    
    df_scaled = df_scaled.copy()
    df_scaled['Cluster_KMeans'] = kmeans.fit_predict(df_scaled)
    
    print(f"✅ K-Means clustering completed.")
    print(f"   Cluster distribution:\n{df_scaled['Cluster_KMeans'].value_counts().sort_index()}")
    
    return df_scaled, kmeans


def apply_dbscan(df_scaled: pd.DataFrame, eps: float = 1.5,
                 min_samples: int = 10) -> Tuple[pd.DataFrame, DBSCAN]:
    """
    Apply DBSCAN clustering to detect arbitrary-shaped clusters and noise.
    
    Args:
        df_scaled: Scaled feature DataFrame
        eps: Maximum distance between samples in the same neighborhood
        min_samples: Minimum number of samples in a neighborhood
        
    Returns:
        Tuple of (DataFrame with cluster labels, fitted DBSCAN model)
    """
    print(f"\n🔹 Applying DBSCAN (eps={eps}, min_samples={min_samples})...")
    print("   This may take a while for large datasets...")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    df_scaled = df_scaled.copy()
    df_scaled['Cluster_DBSCAN'] = dbscan.fit_predict(df_scaled)
    
    n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    n_noise = list(dbscan.labels_).count(-1)
    
    print(f"✅ DBSCAN completed.")
    print(f"   Found {n_clusters} clusters")
    print(f"   Noise points: {n_noise} ({n_noise/len(df_scaled)*100:.2f}%)")
    
    return df_scaled, dbscan


def apply_hierarchical_clustering(df_scaled: pd.DataFrame, 
                                  n_clusters: Optional[int] = None,
                                  sample_size: int = 1000,
                                  save_plot: bool = True,
                                  output_dir: str = "Data/Processed") -> Tuple[pd.DataFrame, AgglomerativeClustering]:
    """
    Apply Hierarchical Clustering (Agglomerative).
    
    Note: For large datasets, this is computationally expensive.
    A sample is used for dendrogram visualization.
    
    Args:
        df_scaled: Scaled feature DataFrame
        n_clusters: Number of clusters. If None, only dendrogram is created.
        sample_size: Number of samples to use for dendrogram
        save_plot: Whether to save dendrogram plot
        output_dir: Directory to save plots
        
    Returns:
        Tuple of (DataFrame with cluster labels, fitted model)
    """
    print(f"\n🔹 Generating Hierarchical Clustering...")
    
    # Create dendrogram on sample
    if len(df_scaled) > sample_size:
        print(f"   Using sample of {sample_size} for dendrogram visualization...")
        subset = df_scaled.sample(sample_size, random_state=42)
    else:
        subset = df_scaled
    
    Z = linkage(subset, method='ward')
    
    plt.figure(figsize=(12, 6))
    dendrogram(
        Z, 
        truncate_mode='level', 
        p=5,
        leaf_rotation=90,
        leaf_font_size=8
    )
    plt.title(f"Hierarchical Clustering Dendrogram (sample of {len(subset)} songs)", 
              fontsize=14, pad=12)
    plt.xlabel("Sample Songs", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.tight_layout()
    
    if save_plot:
        dendro_path = os.path.join(output_dir, "dendrogram.png")
        plt.savefig(dendro_path, dpi=300, bbox_inches='tight')
        print(f"✅ Dendrogram saved to: {dendro_path}")
    
    plt.show()
    
    # Apply clustering to full dataset if n_clusters is specified
    df_scaled = df_scaled.copy()
    
    if n_clusters is not None:
        print(f"   Applying Agglomerative Clustering with {n_clusters} clusters...")
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        df_scaled['Cluster_Hierarchical'] = hierarchical.fit_predict(df_scaled)
        print(f"✅ Hierarchical clustering completed.")
    else:
        hierarchical = None
        print("ℹ️ Dendrogram created. Set n_clusters to apply clustering to full dataset.")
    
    return df_scaled, hierarchical


def clustering_pipeline(scaled_data_path: str, 
                       output_path: str = "Data/Processed/clustered_songs.csv",
                       use_kmeans: bool = True,
                       use_dbscan: bool = True,
                       use_hierarchical: bool = False,
                       optimal_k: Optional[int] = None) -> pd.DataFrame:
    """
    Complete clustering pipeline applying multiple algorithms.
    
    Args:
        scaled_data_path: Path to scaled features CSV
        output_path: Path to save clustered results
        use_kmeans: Whether to apply K-Means
        use_dbscan: Whether to apply DBSCAN
        use_hierarchical: Whether to apply Hierarchical clustering
        optimal_k: Pre-determined optimal k. If None, will be calculated.
        
    Returns:
        DataFrame with cluster labels from all applied algorithms
    """
    # Load scaled data
    print("\n🔹 Loading scaled dataset...")
    if not os.path.exists(scaled_data_path):
        raise FileNotFoundError(f"Scaled data file not found: {scaled_data_path}")
    
    df_scaled = pd.read_csv(scaled_data_path)
    print(f"✅ Loaded dataset with shape: {df_scaled.shape}")
    
    # K-Means Clustering
    if use_kmeans:
        if optimal_k is None:
            optimal_k = find_optimal_k_elbow_silhouette(df_scaled)
        
        df_scaled, kmeans_model = apply_kmeans(df_scaled, optimal_k)
    
    # DBSCAN Clustering
    if use_dbscan:
        df_scaled, dbscan_model = apply_dbscan(df_scaled)
    
    # Hierarchical Clustering (optional, computationally expensive)
    if use_hierarchical:
        df_scaled, hierarchical_model = apply_hierarchical_clustering(df_scaled)
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_scaled.to_csv(output_path, index=False)
    print(f"\n✅ Clustering completed successfully.")
    print(f"📁 Results saved to: {output_path}")
    
    return df_scaled


if __name__ == "__main__":
    # Main execution
    scaled_data_path = "Data/Processed/scaled_features.csv"
    
    try:
        df_clustered = clustering_pipeline(
            scaled_data_path,
            use_kmeans=True,
            use_dbscan=True,
            use_hierarchical=False  # Set to True if you want hierarchical clustering
        )
    except Exception as e:
        print(f"❌ Error during clustering: {e}")
        raise
