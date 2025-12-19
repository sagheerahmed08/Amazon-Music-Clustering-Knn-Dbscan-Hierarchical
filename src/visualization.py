"""
Visualization Module for Amazon Music Clustering Project.

This module creates comprehensive visualizations of clustering results:
- PCA and t-SNE 2D projections
- Feature comparisons across clusters
- Distribution plots
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Optional


def visualize_pca_clusters(df: pd.DataFrame, features: List[str],
                          cluster_column: str = "Cluster_KMeans",
                          save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create PCA 2D visualization with cluster coloring.
    
    Args:
        df: DataFrame with features and cluster labels
        features: List of feature names
        cluster_column: Name of cluster column
        save_path: Optional path to save the figure
        
    Returns:
        DataFrame with PCA components added
    """
    print("\n🔹 Generating PCA 2D visualization with clusters...")
    
    # Filter features to only those that exist
    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        raise ValueError(f"No valid features found. Available columns: {df.columns.tolist()}")
    
    X = df[valid_features].values
    labels = df[cluster_column].values
    
    # Perform PCA
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(X)
    
    # Add PCA components to dataframe
    df_viz = df.copy()
    df_viz['PCA1'] = pca_result[:, 0]
    df_viz['PCA2'] = pca_result[:, 1]
    
    # Explained variance
    explained_var = pca.explained_variance_ratio_
    total_var = sum(explained_var) * 100
    
    # Create visualization
    plt.figure(figsize=(10, 7))
    unique_labels = sorted(df_viz[cluster_column].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = df_viz[cluster_column] == label
        plt.scatter(
            df_viz.loc[mask, 'PCA1'],
            df_viz.loc[mask, 'PCA2'],
            c=[colors[i]],
            label=f'Cluster {label}',
            alpha=0.6,
            s=20,
            edgecolors='k',
            linewidths=0.1
        )
    
    plt.xlabel(f"PC1 ({explained_var[0]*100:.2f}% variance)", fontsize=12)
    plt.ylabel(f"PC2 ({explained_var[1]*100:.2f}% variance)", fontsize=12)
    plt.title(f"PCA 2D Visualization of {cluster_column}\n"
              f"(Total Variance Explained: {total_var:.2f}%)", 
              fontsize=14, pad=12)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ PCA visualization saved to: {save_path}")
    
    plt.show()
    
    print(f"   Total variance explained: {total_var:.2f}%")
    
    return df_viz


def visualize_tsne_clusters(df: pd.DataFrame, features: List[str],
                           cluster_column: str = "Cluster_KMeans",
                           sample_size: int = 5000,
                           save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create t-SNE 2D visualization with cluster coloring.
    
    Args:
        df: DataFrame with features and cluster labels
        features: List of feature names
        cluster_column: Name of cluster column
        sample_size: Number of samples to use (for performance)
        save_path: Optional path to save the figure
        
    Returns:
        DataFrame with t-SNE components added
    """
    print(f"\n🔹 Generating t-SNE 2D visualization (sampling {sample_size} for speed)...")
    
    # Filter features to only those that exist
    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        raise ValueError(f"No valid features found. Available columns: {df.columns.tolist()}")
    
    # Sample data if needed
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"   Using {sample_size} samples from {len(df)} total songs")
    else:
        df_sample = df.copy()
    
    X = df_sample[valid_features].values
    labels = df_sample[cluster_column].values
    
    # Perform t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    tsne_result = tsne.fit_transform(X)
    
    # Create visualization
    plt.figure(figsize=(10, 7))
    unique_labels = sorted(df_sample[cluster_column].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = df_sample[cluster_column] == label
        plt.scatter(
            tsne_result[mask, 0],
            tsne_result[mask, 1],
            c=[colors[i]],
            label=f'Cluster {label}',
            alpha=0.7,
            s=20,
            edgecolors='k',
            linewidths=0.1
        )
    
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.title(f"t-SNE 2D Visualization of {cluster_column}\n"
              f"(Sampled {len(df_sample)} songs)", 
              fontsize=14, pad=12)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ t-SNE visualization saved to: {save_path}")
    
    plt.show()
    
    return df_sample


def visualize_feature_comparison(df: pd.DataFrame, features: List[str],
                                cluster_column: str = "Cluster_KMeans",
                                save_path: Optional[str] = None) -> None:
    """
    Create bar chart comparing average feature values across clusters.
    
    Args:
        df: DataFrame with features and cluster labels
        features: List of feature names
        cluster_column: Name of cluster column
        save_path: Optional path to save the figure
    """
    print("\n🔹 Generating average feature comparison per cluster...")
    
    # Filter features to only those that exist
    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        raise ValueError(f"No valid features found. Available columns: {df.columns.tolist()}")
    
    # Calculate cluster means
    cluster_summary = df.groupby(cluster_column)[valid_features].mean().reset_index()
    cluster_summary_melted = cluster_summary.melt(
        id_vars=cluster_column,
        var_name="Feature",
        value_name="Mean"
    )
    
    # Create visualization
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=cluster_summary_melted,
        x="Feature",
        y="Mean",
        hue=cluster_column,
        palette="Set2"
    )
    plt.title("Average Feature Values per Cluster", fontsize=14, pad=12)
    plt.xlabel("Feature", fontsize=12)
    plt.ylabel("Mean Value", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature comparison chart saved to: {save_path}")
    
    plt.show()


def visualize_feature_distributions(df: pd.DataFrame, features: List[str],
                                    cluster_column: str = "Cluster_KMeans",
                                    max_features: int = 5,
                                    save_dir: Optional[str] = None) -> None:
    """
    Create distribution plots (KDE) for features across clusters.
    
    Args:
        df: DataFrame with features and cluster labels
        features: List of feature names to plot
        cluster_column: Name of cluster column
        max_features: Maximum number of features to plot
        save_dir: Optional directory to save figures
    """
    print(f"\n🔹 Plotting feature distributions across clusters...")
    print(f"   Plotting {min(max_features, len(features))} features...")
    
    features_to_plot = features[:max_features]
    
    for feature in features_to_plot:
        plt.figure(figsize=(10, 5))
        sns.kdeplot(
            data=df,
            x=feature,
            hue=cluster_column,
            fill=True,
            common_norm=False,
            palette="Set2",
            alpha=0.6,
            linewidth=2
        )
        plt.title(f"Distribution of {feature.capitalize()} Across Clusters", 
                 fontsize=14, pad=12)
        plt.xlabel(feature.capitalize(), fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(title="Cluster", fontsize=9)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f"distribution_{feature}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: {save_path}")
        
        plt.show()


def visualization_pipeline(clustered_data_path: str,
                           features: List[str],
                           cluster_column: str = "Cluster_KMeans",
                           output_dir: str = "Data/Processed") -> None:
    """
    Complete visualization pipeline: all cluster visualizations.
    
    Args:
        clustered_data_path: Path to clustered dataset CSV
        features: List of feature names
        cluster_column: Name of cluster column to visualize
        output_dir: Directory to save visualizations
    """
    # Load clustered data
    print("\n🔹 Loading clustered dataset...")
    if not os.path.exists(clustered_data_path):
        raise FileNotFoundError(f"Clustered data file not found: {clustered_data_path}")
    
    df = pd.read_csv(clustered_data_path)
    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # PCA Visualization
    df_pca = visualize_pca_clusters(
        df,
        features,
        cluster_column,
        save_path=os.path.join(output_dir, "pca_clusters.png")
    )
    
    # t-SNE Visualization
    df_tsne = visualize_tsne_clusters(
        df,
        features,
        cluster_column,
        save_path=os.path.join(output_dir, "tsne_clusters.png")
    )
    
    # Feature Comparison
    visualize_feature_comparison(
        df,
        features,
        cluster_column,
        save_path=os.path.join(output_dir, "feature_comparison.png")
    )
    
    # Feature Distributions
    visualize_feature_distributions(
        df,
        features,
        cluster_column,
        max_features=5,
        save_dir=output_dir
    )
    
    print("\n✅ Visualization Completed Successfully!")


if __name__ == "__main__":
    # Main execution
    clustered_data_path = "Data/Processed/clustered_songs.csv"
    
    # Load data to get actual features (after transformations)
    df_temp = pd.read_csv(clustered_data_path)
    features = [col for col in df_temp.columns 
                if not col.startswith('Cluster_') and col not in ['PCA1', 'PCA2', 'TSNE1', 'TSNE2']]
    
    try:
        visualization_pipeline(
            clustered_data_path,
            features,
            cluster_column="Cluster_KMeans"
        )
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        raise
