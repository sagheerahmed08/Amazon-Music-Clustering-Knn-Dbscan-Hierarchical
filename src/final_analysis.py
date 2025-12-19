"""
Final Analysis Module for Amazon Music Clustering Project.

This module:
- Combines original data with cluster labels
- Generates top tracks per cluster
- Creates comprehensive summary reports
- Exports final datasets
"""

import os
import pandas as pd
from typing import List, Optional, Dict


def merge_clusters_with_original(original_data_path: str,
                                 clustered_data_path: str,
                                 cluster_columns: List[str] = None) -> pd.DataFrame:
    """
    Merge cluster labels with original dataset.
    
    Args:
        original_data_path: Path to original raw dataset
        clustered_data_path: Path to clustered dataset with labels
        cluster_columns: List of cluster column names to merge
        
    Returns:
        DataFrame with original data and cluster labels
    """
    print("\n🔹 Loading datasets...")
    
    if not os.path.exists(original_data_path):
        raise FileNotFoundError(f"Original data file not found: {original_data_path}")
    if not os.path.exists(clustered_data_path):
        raise FileNotFoundError(f"Clustered data file not found: {clustered_data_path}")
    
    original_df = pd.read_csv(original_data_path)
    clustered_df = pd.read_csv(clustered_data_path)
    
    print(f"✅ Original data shape: {original_df.shape}")
    print(f"✅ Clustered data shape: {clustered_df.shape}")
    
    # Validate shapes match
    if len(original_df) != len(clustered_df):
        raise ValueError(
            f"Data shape mismatch: original={len(original_df)}, "
            f"clustered={len(clustered_df)}"
        )
    
    # Merge cluster labels
    final_df = original_df.copy()
    
    if cluster_columns is None:
        # Auto-detect cluster columns
        cluster_columns = [col for col in clustered_df.columns 
                          if col.startswith('Cluster_')]
    
    for col in cluster_columns:
        if col in clustered_df.columns:
            final_df[col] = clustered_df[col].values
            print(f"   ✅ Added {col} to final dataset")
        else:
            print(f"   ⚠️ Warning: {col} not found in clustered data")
    
    return final_df


def get_top_tracks_per_cluster(df: pd.DataFrame,
                               cluster_column: str = "Cluster_KMeans",
                               popularity_column: str = "popularity_songs",
                               top_n: int = 5,
                               track_name_col: str = "name_song",
                               artist_name_col: str = "name_artists") -> pd.DataFrame:
    """
    Extract top N tracks per cluster based on popularity.
    
    Args:
        df: DataFrame with cluster labels and track information
        cluster_column: Name of cluster column
        popularity_column: Name of popularity column
        top_n: Number of top tracks per cluster
        track_name_col: Name of track name column
        artist_name_col: Name of artist name column
        
    Returns:
        DataFrame with top tracks per cluster
    """
    print(f"\n🔹 Extracting Top {top_n} songs per {cluster_column}...")
    
    # Check if required columns exist
    required_cols = [cluster_column, popularity_column, track_name_col, artist_name_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Group by cluster and get top N tracks
    top_tracks = (
        df.groupby(cluster_column)
        .apply(
            lambda x: x.nlargest(top_n, popularity_column)[
                [track_name_col, artist_name_col, popularity_column, cluster_column]
            ]
        )
        .reset_index(drop=True)
    )
    
    # Sort by cluster and popularity
    top_tracks = top_tracks.sort_values([cluster_column, popularity_column], 
                                        ascending=[True, False])
    
    print(f"\nTop Tracks Summary:")
    for cluster_id in sorted(top_tracks[cluster_column].unique()):
        cluster_tracks = top_tracks[top_tracks[cluster_column] == cluster_id]
        print(f"\n  Cluster {cluster_id}:")
        for idx, row in cluster_tracks.iterrows():
            print(f"    - {row[track_name_col]} by {row[artist_name_col]} "
                  f"(Popularity: {row[popularity_column]})")
    
    return top_tracks


def generate_cluster_report(df: pd.DataFrame,
                           cluster_summary: pd.DataFrame,
                           features: List[str],
                           cluster_column: str = "Cluster_KMeans") -> str:
    """
    Generate a comprehensive text report summarizing cluster analysis.
    
    Args:
        df: DataFrame with cluster labels
        cluster_summary: DataFrame with cluster-wise feature averages
        features: List of feature names
        cluster_column: Name of cluster column
        
    Returns:
        String containing the report
    """
    print("\n🔹 Generating summary report...")
    
    # Filter features to only those in cluster_summary
    valid_features = [f for f in features if f in cluster_summary.columns]
    
    report_lines = []
    report_lines.append("🎵 Amazon Music Clustering — Final Summary Report\n")
    report_lines.append("=" * 60 + "\n\n")
    report_lines.append(f"Total songs analyzed: {len(df):,}\n")
    report_lines.append(f"Number of {cluster_column} clusters: "
                       f"{df[cluster_column].nunique()}\n")
    report_lines.append("=" * 60 + "\n\n")
    
    # Cluster size distribution
    cluster_sizes = df[cluster_column].value_counts().sort_index()
    report_lines.append("📊 Cluster Size Distribution:\n")
    for cluster_id, size in cluster_sizes.items():
        percentage = (size / len(df)) * 100
        report_lines.append(f"   Cluster {cluster_id}: {size:,} songs ({percentage:.2f}%)\n")
    report_lines.append("\n")
    
    # Detailed cluster analysis
    report_lines.append("🎧 Detailed Cluster Analysis:\n")
    report_lines.append("-" * 60 + "\n")
    
    for cluster_id, row in cluster_summary.iterrows():
        report_lines.append(f"\n🎧 Cluster {cluster_id} Summary:\n")
        report_lines.append(f"   Size: {cluster_sizes[cluster_id]:,} songs\n")
        
        # High and low features (only from valid features)
        cluster_mean = row.mean()
        high_feats = [f for f in row[row > cluster_mean].index.tolist() if f in valid_features]
        low_feats = [f for f in row[row < cluster_mean].index.tolist() if f in valid_features]
        
        report_lines.append(f"   🔺 High in: {', '.join(high_feats[:8])}\n")
        if len(high_feats) > 8:
            report_lines.append(f"      ... and {len(high_feats) - 8} more features\n")
        
        report_lines.append(f"   🔻 Low in: {', '.join(low_feats[:8])}\n")
        if len(low_feats) > 8:
            report_lines.append(f"      ... and {len(low_feats) - 8} more features\n")
        
        # Interpretation
        interpretation = []
        if 'energy' in high_feats and 'danceability' in high_feats:
            interpretation.append("Upbeat / Party tracks")
        if 'acousticness' in high_feats:
            interpretation.append("Chill or Acoustic songs")
        if 'instrumentalness' in high_feats:
            interpretation.append("Instrumental / Ambient tracks")
        if 'valence' in high_feats:
            interpretation.append("Positive mood")
        if 'speechiness' in high_feats:
            interpretation.append("Speech-heavy / Rap")
        if 'liveness' in high_feats:
            interpretation.append("Live performance")
        if not interpretation:
            interpretation.append("Mixed characteristics")
        
        report_lines.append(f"   💡 Possible interpretation: {', '.join(interpretation)}\n")
        
        # Key feature values
        report_lines.append(f"   📈 Key Feature Values:\n")
        key_features = ['danceability', 'energy', 'valence', 'tempo']
        for feat in key_features:
            if feat in row.index:
                report_lines.append(f"      {feat}: {row[feat]:.3f}\n")
    
    report_lines.append("\n" + "=" * 60 + "\n")
    report_lines.append("Report generated successfully.\n")
    
    report_text = "".join(report_lines)
    return report_text


def final_analysis_pipeline(original_data_path: str,
                            clustered_data_path: str,
                            cluster_summary_path: str,
                            features: List[str],
                            cluster_column: str = "Cluster_KMeans",
                            output_dir: str = "Data/Processed") -> Dict[str, pd.DataFrame]:
    """
    Complete final analysis pipeline.
    
    Args:
        original_data_path: Path to original dataset
        clustered_data_path: Path to clustered dataset
        cluster_summary_path: Path to cluster summary CSV
        features: List of feature names
        cluster_column: Name of cluster column
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary containing final datasets
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Merge clusters with original data
    final_df = merge_clusters_with_original(
        original_data_path,
        clustered_data_path,
        cluster_columns=[cluster_column]
    )
    
    # 2. Load cluster summary
    if os.path.exists(cluster_summary_path):
        cluster_summary = pd.read_csv(cluster_summary_path, index_col=0)
    else:
        print("⚠️ Cluster summary not found. Generating from final dataset...")
        cluster_summary = final_df.groupby(cluster_column)[features].mean()
    
    # 3. Get top tracks per cluster
    top_tracks = get_top_tracks_per_cluster(final_df, cluster_column)
    
    # Save top tracks
    top_tracks_path = os.path.join(output_dir, "top_tracks_per_cluster.csv")
    top_tracks.to_csv(top_tracks_path, index=False)
    print(f"✅ Top tracks saved to: {top_tracks_path}")
    
    # 4. Generate report
    report_text = generate_cluster_report(final_df, cluster_summary, features, cluster_column)
    
    report_path = os.path.join(output_dir, "final_cluster_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"✅ Summary report saved to: {report_path}")
    
    # 5. Export final dataset
    final_export_path = os.path.join(output_dir, "final_clustered_dataset.csv")
    final_df.to_csv(final_export_path, index=False)
    print(f"✅ Final dataset exported to: {final_export_path}")
    
    print("\n🎉 Final Analysis Completed — Project Ready for Presentation!")
    
    return {
        'final_dataset': final_df,
        'top_tracks': top_tracks,
        'cluster_summary': cluster_summary
    }


if __name__ == "__main__":
    # Main execution
    original_data_path = "Data/Raw/single_genre_artists.csv"
    clustered_data_path = "Data/Processed/clustered_songs.csv"
    cluster_summary_path = "Data/Processed/cluster_summary.csv"
    
    # Load clustered data to get actual features (after transformations)
    clustered_df_temp = pd.read_csv(clustered_data_path)
    features = [col for col in clustered_df_temp.columns 
                if not col.startswith('Cluster_') and col not in ['PCA1', 'PCA2', 'TSNE1', 'TSNE2']]
    
    try:
        results = final_analysis_pipeline(
            original_data_path,
            clustered_data_path,
            cluster_summary_path,
            features,
            cluster_column="Cluster_KMeans"
        )
    except Exception as e:
        print(f"❌ Error during final analysis: {e}")
        raise
