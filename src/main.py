"""
Main Entry Point for Amazon Music Clustering Project.

This script orchestrates the complete pipeline:
1. Data Preprocessing
2. Clustering
3. Evaluation
4. Visualization
5. Final Analysis

Usage:
    python src/main.py
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import preprocess_pipeline
from clustering import clustering_pipeline
from evaluation import evaluation_pipeline
from visualization import visualization_pipeline
from final_analysis import final_analysis_pipeline


# Configuration
# Note: Features will be dynamically extracted from preprocessed data
# to handle transformations (log transform, one-hot encoding)
DATA_DIR = "Data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "Raw", "single_genre_artists.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "Processed")
SCALED_DATA_PATH = os.path.join(PROCESSED_DIR, "scaled_features.csv")
CLUSTERED_DATA_PATH = os.path.join(PROCESSED_DIR, "clustered_songs.csv")
CLUSTER_SUMMARY_PATH = os.path.join(PROCESSED_DIR, "cluster_summary.csv")


def main():
    """Execute the complete clustering pipeline."""
    print("=" * 70)
    print("🎵 Amazon Music Clustering - Complete Pipeline")
    print("=" * 70)
    
    try:
        # Step 1: Preprocessing
        print("\n" + "=" * 70)
        print("STEP 1: DATA PREPROCESSING")
        print("=" * 70)
        df_original, df_scaled, features = preprocess_pipeline(
            RAW_DATA_PATH,
            output_dir=PROCESSED_DIR,
            visualize=True
        )
        
        # Step 2: Clustering
        print("\n" + "=" * 70)
        print("STEP 2: CLUSTERING")
        print("=" * 70)
        df_clustered = clustering_pipeline(
            SCALED_DATA_PATH,
            output_path=CLUSTERED_DATA_PATH,
            use_kmeans=True,
            use_dbscan=True,
            use_hierarchical=False  # Set to True if needed (computationally expensive)
        )
        
        # Step 3: Evaluation
        # Load clustered data to get actual feature names (after transformations)
        df_clustered_temp = pd.read_csv(CLUSTERED_DATA_PATH)
        # Extract feature columns (exclude cluster columns)
        actual_features = [col for col in df_clustered_temp.columns 
                          if not col.startswith('Cluster_') and col not in ['PCA1', 'PCA2', 'TSNE1', 'TSNE2']]
        
        print("\n" + "=" * 70)
        print("STEP 3: CLUSTER EVALUATION")
        print("=" * 70)
        metrics, cluster_summary = evaluation_pipeline(
            CLUSTERED_DATA_PATH,
            actual_features,
            cluster_column="Cluster_KMeans",
            output_dir=PROCESSED_DIR
        )
        
        # Step 4: Visualization
        print("\n" + "=" * 70)
        print("STEP 4: VISUALIZATION")
        print("=" * 70)
        visualization_pipeline(
            CLUSTERED_DATA_PATH,
            actual_features,
            cluster_column="Cluster_KMeans",
            output_dir=PROCESSED_DIR
        )
        
        # Step 5: Final Analysis
        print("\n" + "=" * 70)
        print("STEP 5: FINAL ANALYSIS")
        print("=" * 70)
        results = final_analysis_pipeline(
            RAW_DATA_PATH,
            CLUSTERED_DATA_PATH,
            CLUSTER_SUMMARY_PATH,
            actual_features,
            cluster_column="Cluster_KMeans",
            output_dir=PROCESSED_DIR
        )
        
        # Final Summary
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n📊 Results Summary:")
        print(f"   - Total songs analyzed: {len(df_original):,}")
        print(f"   - Number of clusters: {metrics.get('n_clusters', 'N/A')}")
        print(f"   - Silhouette Score: {metrics.get('silhouette_score', 0):.4f}")
        print(f"   - Davies-Bouldin Index: {metrics.get('davies_bouldin_index', 0):.4f}")
        print(f"\n📁 Output Files:")
        print(f"   - Scaled features: {SCALED_DATA_PATH}")
        print(f"   - Clustered data: {CLUSTERED_DATA_PATH}")
        print(f"   - Cluster summary: {CLUSTER_SUMMARY_PATH}")
        print(f"   - Final dataset: {os.path.join(PROCESSED_DIR, 'final_clustered_dataset.csv')}")
        print(f"   - Summary report: {os.path.join(PROCESSED_DIR, 'final_cluster_report.txt')}")
        print("\n🎉 Project ready for presentation!")
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("Please ensure the data file exists at the specified path.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

