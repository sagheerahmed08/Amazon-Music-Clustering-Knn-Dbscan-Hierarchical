"""
Data Preprocessing Module for Amazon Music Clustering Project.

This module handles:
- Data loading and exploration
- Missing value and duplicate handling
- Feature selection
- Data normalization/scaling
- Dimensionality reduction (PCA, t-SNE) for visualization
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Tuple, List, Optional


# Configuration constants
FEATURES = [
        'duration_ms', 'danceability', 'energy',
        'loudness', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo'
    ]

COLUMNS_TO_DROP = [ 
    'id_songs', 'name_song', 'id_artists', 
    'release_date', 'genres', 'name_artists'
]


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"Dataset is empty: {file_path}")
        return df
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"File is empty: {file_path}")


def explore_data(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Perform basic data exploration and return summary statistics.
    
    Args:
        df: Input DataFrame
        verbose: If True, print exploration results
        
    Returns:
        Dictionary containing exploration statistics
    """
    stats = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'numeric_summary': df.describe().T.to_dict()
    }
    
    if verbose:
        print("\nFirst five rows of the dataset:")
        print(df.head())
        print(f"\nShape of dataset: {df.shape}")
        print(f"\nColumn names:\n{df.columns.tolist()}")
        print(f"\nData types:\n{df.dtypes}")
        print("\nInfo Summary:")
        print(df.info())
        print("\nStatistical Summary:")
        print(df.describe().T)
        print(f"\nMissing values per column:\n{df.isnull().sum()}")
        print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
    
    return stats


def clean_data(df: pd.DataFrame, columns_to_drop: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove unnecessary columns from the dataset.
    
    Args:
        df: Input DataFrame
        columns_to_drop: List of column names to drop. If None, uses default list.
        
    Returns:
        DataFrame with specified columns removed
    """
    if columns_to_drop is None:
        columns_to_drop = COLUMNS_TO_DROP
    
    df_cleaned = df.copy()
    
    print(f"\nColumns before dropping: {len(df_cleaned.columns)}")
    print(f"Columns: {df_cleaned.columns.tolist()}")
    
    # Drop columns that exist
    existing_cols = [col for col in columns_to_drop if col in df_cleaned.columns]
    df_cleaned.drop(columns=existing_cols, inplace=True, errors='ignore')
    
    print(f"\nColumns after dropping: {len(df_cleaned.columns)}")
    print(f"Columns: {df_cleaned.columns.tolist()}")
    
    return df_cleaned


def validate_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    """
    Validate that all required features exist in the DataFrame.
    
    Args:
        df: Input DataFrame
        features: List of feature names to validate
        
    Returns:
        List of valid features that exist in the DataFrame
        
    Raises:
        ValueError: If no valid features are found
    """
    valid_features = [f for f in features if f in df.columns]
    
    if not valid_features:
        raise ValueError("No valid features found in the dataset")
    
    missing_features = set(features) - set(valid_features)
    if missing_features:
        print(f"[WARNING] Missing features: {missing_features}")
    
    return valid_features


def visualize_feature_distributions(df: pd.DataFrame, features: List[str], 
                                    save_path: Optional[str] = None) -> None:
    """
    Create histograms for feature distributions.
    
    Args:
        df: Input DataFrame
        features: List of feature names to visualize
        save_path: Optional path to save the figure
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    for i, col in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(col.capitalize(), fontsize=12)
        plt.xlabel(col)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature distribution plot saved to: {save_path}")
    
    plt.show()


def visualize_correlation_heatmap(df: pd.DataFrame, features: List[str],
                                  save_path: Optional[str] = None) -> None:
    """
    Create a correlation heatmap for features.
    
    Args:
        df: Input DataFrame
        features: List of feature names
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    correlation_matrix = df[features].corr()
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f",
        center=0,
        square=True,
        linewidths=0.5
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, pad=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to: {save_path}")
    
    plt.show()


def transform_features(df: pd.DataFrame, 
                      log_features: List[str] = None,
                      categorical_features: List[str] = None) -> pd.DataFrame:
    """
    Apply feature transformations based on distribution characteristics.
    
    Steps:
    1. Log transform right-skewed features
    2. One-hot encode categorical features
    
    Args:
        df: Input DataFrame
        log_features: List of features needing log transformation
        categorical_features: List of categorical features for one-hot encoding
        
    Returns:
        Transformed DataFrame
    """
    df_transformed = df.copy()
    
    # Default features based on distribution analysis
    if log_features is None:
        log_features = ['duration_ms', 'speechiness', 'instrumentalness', 'liveness']
        # Optional: 'acousticness' (mild skew) - can be added if needed
        # Optional: 'followers' if it exists in the dataset
    
    if categorical_features is None:
        categorical_features = ['key', 'mode', 'time_signature']
    
    # Step 1: Apply log transformation to right-skewed features
    print("\n Applying log transformation to skewed features...")
    for feature in log_features:
        if feature in df_transformed.columns:
            # Use log1p to handle zeros: log(1 + x)
            # This is safer than log() as it handles 0 values gracefully
            df_transformed[feature] = np.log1p(df_transformed[feature])
            print(f"  Log transformed: {feature}")
        else:
            print(f"  Feature not found: {feature}")
    
    # Step 2: One-hot encode categorical features
    print(f"\n One-hot encoding categorical features...")
    for feature in categorical_features:
        if feature in df_transformed.columns:
            # Create dummy variables
            dummies = pd.get_dummies(
                df_transformed[feature], 
                prefix=feature, 
                drop_first=False
            )
            # Concatenate with main dataframe
            df_transformed = pd.concat([df_transformed, dummies], axis=1)
            # Drop original column
            df_transformed.drop(columns=[feature], inplace=True)
            print(f"One-hot encoded: {feature} -> {len(dummies.columns)} columns")
        else:
            print(f"Feature not found: {feature}")
    
    print(f"\nFeature transformation completed!")
    print(f"   Shape before: {df.shape}, Shape after: {df_transformed.shape}")
    
    return df_transformed


def scale_features(df: pd.DataFrame, features: List[str], 
                   scaler_type: str = 'standard',
                   apply_transformations: bool = False,
                   log_features: List[str] = None,
                   categorical_features: List[str] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale/normalize features using StandardScaler or MinMaxScaler.
    Optionally applies log transformation and one-hot encoding before scaling.
    
    Args:
        df: Input DataFrame
        features: List of feature names to scale
        scaler_type: Type of scaler ('standard' or 'minmax')
        apply_transformations: If True, applies log transform and one-hot encoding
        log_features: List of features needing log transformation
        categorical_features: List of categorical features for one-hot encoding
        
    Returns:
        Tuple of (scaled DataFrame, fitted scaler object)
    """
    df_processed = df.copy()
    
    # Apply transformations if requested
    if apply_transformations:
        df_processed = transform_features(df_processed, log_features, categorical_features)
        # Update features list to include one-hot encoded columns
        all_features = df_processed.columns.tolist()
        # Filter to only include features that were originally requested or are one-hot encoded
        features = [f for f in all_features if f in features or 
                   any(f.startswith(cat + '_') for cat in (categorical_features or ['key', 'mode', 'time_signature']))]
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    # Only scale features that exist and are numerical
    valid_features = [f for f in features if f in df_processed.columns]
    
    # Filter to only numerical features
    numerical_features = [f for f in valid_features 
                          if df_processed[f].dtype in [np.float64, np.int64, np.float32, np.int32, np.float16, np.int16]]
    
    if numerical_features:
        scaled_data = scaler.fit_transform(df_processed[numerical_features])
        df_processed[numerical_features] = scaled_data
        
        print("\nScaled Data (first 5 rows):")
        print(df_processed[numerical_features].head())
        print("\nScaled Data Summary:")
        print(df_processed[numerical_features].describe().T)
    else:
        print("Warning: No valid numerical features to scale")
    
    return df_processed, scaler


def perform_pca(df_scaled: pd.DataFrame, n_components: int = 2,
                save_path: Optional[str] = None) -> Tuple[pd.DataFrame, PCA]:
    """
    Perform Principal Component Analysis for dimensionality reduction.
    
    Args:
        df_scaled: Scaled DataFrame
        n_components: Number of principal components
        save_path: Optional path to save the visualization
        
    Returns:
        Tuple of (PCA DataFrame, fitted PCA object)
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(df_scaled)
    
    df_pca = pd.DataFrame(
        pca_result, 
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    explained_var = pca.explained_variance_ratio_
    print(f"\n PCA Explained Variance Ratio: {explained_var}")
    print(f"Total Variance Retained: {sum(explained_var)*100:.2f}%")
    
    # Visualize PCA
    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca['PC1'], df_pca['PC2'], s=2, alpha=0.5, color='royalblue')
    plt.title("PCA Projection (2D)", fontsize=14)
    plt.xlabel(f"PC1 ({explained_var[0]*100:.2f}% variance)")
    plt.ylabel(f"PC2 ({explained_var[1]*100:.2f}% variance)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA visualization saved to: {save_path}")
    
    plt.show()
    
    return df_pca, pca


def perform_tsne(df_scaled: pd.DataFrame, n_components: int = 2,
                 perplexity: int = 30, max_iter: int = 1000,
                 sample_size: Optional[int] = None,
                 save_path: Optional[str] = None) -> Tuple[pd.DataFrame, TSNE]:
    """
    Perform t-SNE for dimensionality reduction and visualization.
    
    Args:
        df_scaled: Scaled DataFrame
        n_components: Number of components (typically 2 or 3)
        perplexity: Perplexity parameter for t-SNE
        max_iter: Maximum number of iterations
        sample_size: If provided, sample this many rows for faster computation
        save_path: Optional path to save the visualization
        
    Returns:
        Tuple of (t-SNE DataFrame, fitted t-SNE object)
    """
    data_to_transform = df_scaled
    
    if sample_size and len(df_scaled) > sample_size:
        print(f"Sampling {sample_size} rows for t-SNE (faster computation)")
        data_to_transform = df_scaled.sample(n=sample_size, random_state=42)
    
    tsne = TSNE(
        n_components=n_components, 
        perplexity=perplexity, 
        max_iter=max_iter, 
        random_state=42
    )
    tsne_result = tsne.fit_transform(data_to_transform)
    
    df_tsne = pd.DataFrame(
        tsne_result, 
        columns=[f'TSNE{i+1}' for i in range(n_components)]
    )
    
    # Visualize t-SNE
    plt.figure(figsize=(8, 6))
    plt.scatter(df_tsne['TSNE1'], df_tsne['TSNE2'], s=2, alpha=0.6, color='teal')
    plt.title("t-SNE Visualization (2D)", fontsize=14)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved to: {save_path}")
    
    plt.show()
    
    return df_tsne, tsne


def preprocess_pipeline(data_path: str, output_dir: str = "Data/Processed",
                       visualize: bool = True,
                       apply_transformations: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Complete preprocessing pipeline: load, explore, clean, transform, scale data.
    
    Args:
        data_path: Path to raw data CSV file
        output_dir: Directory to save processed data
        visualize: Whether to generate visualizations
        apply_transformations: Whether to apply log transform and one-hot encoding
        
    Returns:
        Tuple of (original DataFrame, scaled DataFrame, feature list)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    print("Loading dataset...")
    df = load_data(data_path)
    
    # 2. Explore data
    print("\nExploring dataset...")
    explore_data(df, verbose=True)
    
    # 3. Clean data
    print("\nCleaning dataset...")
    df_cleaned = clean_data(df)
    
    # 4. Validate features
    print("\nValidating features...")
    valid_features = validate_features(df_cleaned, FEATURES)
    
    # 5. Visualizations (before transformations)
    if visualize:
        print("\nCreating feature distribution plots (before transformations)...")
        visualize_feature_distributions(
            df_cleaned, 
            valid_features,
            save_path=os.path.join(output_dir, "feature_distributions_before.png")
        )
        
        print("\nCreating correlation heatmap...")
        visualize_correlation_heatmap(
            df_cleaned,
            valid_features,
            save_path=os.path.join(output_dir, "correlation_heatmap.png")
        )
    
    # 6. Apply transformations (log transform, one-hot encoding)
    if apply_transformations:
        print("\nApplying feature transformations...")
        df_cleaned = transform_features(df_cleaned)
        # After transformations, extract all numerical features (including one-hot encoded)
        # Exclude any non-feature columns that might have been added
        all_numerical = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        # Keep original valid_features plus any new numerical features from transformations
        valid_features = [f for f in all_numerical if f in valid_features or 
                         any(f.startswith(cat + '_') for cat in ['key', 'mode', 'time_signature']) or
                         f in ['popularity_songs', 'duration_ms', 'explicit', 'followers', 'popularity_artists']]
    
    # 7. Scale features
    print("\nScaling features...")
    # Get all numerical features for scaling
    numerical_features = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude any metadata columns if they exist
    exclude_cols = ['id_songs', 'id_artists']  # Add any other non-feature columns
    numerical_features = [f for f in numerical_features if f not in exclude_cols]
    
    df_scaled, scaler = scale_features(
        df_cleaned, 
        numerical_features,
        apply_transformations=False  # Already applied above
    )
    
    # Update valid_features to match what was actually scaled
    valid_features = numerical_features
    
    # Save scaled features
    scaled_path = os.path.join(output_dir, "scaled_features.csv")
    df_scaled.to_csv(scaled_path, index=False)
    print(f"Scaled features saved to: {scaled_path}")
    
    # 7. Dimensionality reduction (for visualization)
    print("\nPerforming PCA...")
    df_pca, pca = perform_pca(
        df_scaled,
        save_path=os.path.join(output_dir, "pca_visualization.png")
    )
    
    print("\nPerforming t-SNE...")
    df_tsne, tsne = perform_tsne(
        df_scaled,
        sample_size=5000,  # Sample for faster computation
        save_path=os.path.join(output_dir, "tsne_visualization.png")
    )
    
    print("\nData Exploration & Preprocessing Completed Successfully!")
    print(f"-> Selected features: {valid_features}")
    print(f"-> Shape of scaled dataset: {df_scaled.shape}")
    
    return df_cleaned, df_scaled, valid_features


if __name__ == "__main__":
    # Main execution
    data_path = "Data/Raw/single_genre_artists.csv"
    
    try:
        df_original, df_scaled, features = preprocess_pipeline(data_path, visualize=True)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise
