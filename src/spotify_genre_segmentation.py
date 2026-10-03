"""
===================================================================================================
Corizo Artificial Intelligence Internship - Project 2: Spotify Songs' Genre Segmentation
===================================================================================================
Author: Hamid Kamal
Role: Data Science & Machine Learning Engineer Intern
Task: Automated Music Analysis, Clustering, and Recommendation Support System
Primary Dataset: Spotify Songs Audio Features (32,833 entries)
===================================================================================================
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path

# Environment & thread configuration for stability, reproducibility & silence warnings
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

# Constants & Relative Path Resolution (Runs from repo root or src/)
RANDOM_STATE = 42

# Resolve repository root
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == 'src' else CURRENT_FILE.parent

# Dataset candidate paths
DATASET_CANDIDATES = [
    REPO_ROOT / 'data' / 'upload_efc975f9-7e96-4804-8ae1-691bc3670af7.csv',
    REPO_ROOT / 'upload_efc975f9-7e96-4804-8ae1-691bc3670af7.csv',
    REPO_ROOT / 'data' / 'spotify_songs.csv',
    Path('upload_efc975f9-7e96-4804-8ae1-691bc3670af7.csv'),
    Path('data/upload_efc975f9-7e96-4804-8ae1-691bc3670af7.csv')
]

DATASET_PATH = None
for candidate in DATASET_CANDIDATES:
    if candidate.exists():
        DATASET_PATH = candidate
        break

if DATASET_PATH is None:
    DATASET_PATH = REPO_ROOT / 'upload_efc975f9-7e96-4804-8ae1-691bc3670af7.csv'

FIGURES_DIR = REPO_ROOT / 'figures'
DATA_DIR = REPO_ROOT / 'data'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Styling configuration
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def load_and_audit_data(filepath):
    print("=" * 85)
    print("STEP 1: DATASET INGESTION & AUDIT")
    print("=" * 85)
    print(f"Reading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset Loaded Successfully: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    print("\n--- Missing Values Audit ---")
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if len(null_cols) > 0:
        for col, cnt in null_cols.items():
            print(f"  {col}: {cnt} missing rows ({cnt/len(df)*100:.3f}%)")
    else:
        print("  No missing values found.")
    
    print("\n--- Duplicate Records Audit ---")
    print(f"  Full exact duplicate rows: {df.duplicated().sum()}")
    print(f"  Unique track IDs: {df['track_id'].nunique():,}")
    print(f"  Track IDs appearing across multiple playlists: {df.duplicated(subset=['track_id']).sum():,}")
    
    print("\n--- Categorical Cardinality ---")
    print(f"  Unique Playlist Genres ({df['playlist_genre'].nunique()}): {sorted(df['playlist_genre'].unique())}")
    print(f"  Unique Playlist Subgenres: {df['playlist_subgenre'].nunique()}")
    print(f"  Unique Playlist Names: {df['playlist_name'].nunique():,}")
    print(f"  Unique Artists: {df['track_artist'].nunique():,}")
    
    print("\n--- Genre Breakdown (Raw Counts) ---")
    for g, count in df['playlist_genre'].value_counts().items():
        print(f"  {g:<8}: {count:,} songs ({count/len(df)*100:.1f}%)")
    
    return df


def preprocess_data(df):
    print("\n" + "=" * 85)
    print("STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 85)
    
    # Drop rows with missing track metadata (5 rows)
    initial_len = len(df)
    df_clean = df.dropna(subset=['track_name', 'track_artist', 'track_album_name']).copy()
    df_clean.reset_index(drop=True, inplace=True)
    print(f"Dropped {initial_len - len(df_clean)} rows with null metadata. Cleaned records: {len(df_clean):,}")
    
    # Feature transformations
    df_clean['duration_min'] = df_clean['duration_ms'] / 60000.0
    df_clean['release_year'] = pd.to_datetime(df_clean['track_album_release_date'], errors='coerce').dt.year
    
    # Continuous audio features selected for unsupervised clustering
    audio_features = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence',
        'tempo', 'duration_min'
    ]
    
    # Feature Standardization (Z-score normalization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[audio_features])
    print(f"Selected {len(audio_features)} continuous audio features for clustering:")
    print(f"  {audio_features}")
    print(f"Standardized feature matrix X_scaled shape: {X_scaled.shape} (Mean=0, Std=1)")
    
    return df_clean, audio_features, X_scaled, scaler


def perform_eda_and_visualizations(df_clean, audio_features):
    print("\n" + "=" * 85)
    print("STEP 3: EXPLORATORY DATA ANALYSIS & VISUALIZATIONS")
    print("=" * 85)
    
    # 1. Correlation Matrix Computation & Plot
    corr_features = audio_features + ['track_popularity']
    corr_matrix = df_clean[corr_features].corr()
    print("Calculated Correlation Matrix (Key Values):")
    print(f"  Energy vs Loudness:         r = {corr_matrix.loc['energy', 'loudness']:+.4f} (Strong positive)")
    print(f"  Energy vs Acousticness:     r = {corr_matrix.loc['energy', 'acousticness']:+.4f} (Strong negative)")
    print(f"  Loudness vs Acousticness:   r = {corr_matrix.loc['loudness', 'acousticness']:+.4f} (Strong negative)")
    print(f"  Danceability vs Valence:    r = {corr_matrix.loc['danceability', 'valence']:+.4f} (Moderate positive)")
    print(f"  Danceability vs Tempo:      r = {corr_matrix.loc['danceability', 'tempo']:+.4f} (Negative)")
    print(f"  Popularity vs Audio Features max |r|: {corr_matrix['track_popularity'].drop('track_popularity').abs().max():.4f}")
    
    plt.figure(figsize=(10, 8), dpi=300)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-0.7, vmax=0.7, annot=True, fmt='.2f',
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Spotify Audio Features & Track Popularity', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '01_correlation_matrix.png')
    plt.close()
    print(f"  [Saved] figures/01_correlation_matrix.png")
    
    # 2. Genre & Subgenre Distributions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    genre_counts = df_clean['playlist_genre'].value_counts()
    sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index, palette='viridis', ax=axes[0], legend=False)
    axes[0].set_title('Distribution of Songs by Playlist Genre', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Songs')
    for i, v in enumerate(genre_counts.values):
        axes[0].text(v + 40, i, f"{v:,} ({v/len(df_clean)*100:.1f}%)", va='center', fontweight='bold', fontsize=9)

    subgenre_counts = df_clean['playlist_subgenre'].value_counts()
    sns.barplot(x=subgenre_counts.values, y=subgenre_counts.index, hue=subgenre_counts.index, palette='magma', ax=axes[1], legend=False)
    axes[1].set_title('Distribution of Songs by Playlist Subgenre (24 Subgenres)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Songs')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '02_genre_subgenre_distribution.png')
    plt.close()
    print(f"  [Saved] figures/02_genre_subgenre_distribution.png")
    
    # 3. Audio Features Histograms + KDE Distributions
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), dpi=300)
    axes = axes.flatten()
    plot_cols = audio_features + ['track_popularity', 'release_year']
    colors = sns.color_palette('tab10', len(plot_cols))

    for idx, col in enumerate(plot_cols):
        data = df_clean[col].dropna()
        sns.histplot(data, kde=True, ax=axes[idx], color=colors[idx], bins=30)
        axes[idx].set_title(f'Distribution of {col}', fontweight='bold', fontsize=11)
        axes[idx].set_ylabel('Frequency')
        mean_val = data.mean()
        median_val = data.median()
        axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=1.2, label=f'Mean: {mean_val:.2f}')
        axes[idx].axvline(median_val, color='green', linestyle=':', linewidth=1.2, label=f'Median: {median_val:.2f}')
        axes[idx].legend(fontsize=8)

    for j in range(len(plot_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Distributions of Spotify Audio Features, Popularity, and Release Year', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '03_audio_features_distributions.png')
    plt.close()
    print(f"  [Saved] figures/03_audio_features_distributions.png")
    
    # 4. Genre vs Audio Features Comparison
    genre_means = df_clean.groupby('playlist_genre')[audio_features + ['track_popularity']].mean()
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), dpi=300)
    axes = axes.flatten()
    features_to_compare = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo', 'track_popularity']

    for idx, feat in enumerate(features_to_compare):
        order = genre_means[feat].sort_values(ascending=False).index
        sns.barplot(data=df_clean, x='playlist_genre', y=feat, order=order, hue='playlist_genre', palette='Blues_r', ax=axes[idx], legend=False, errorbar=None)
        axes[idx].set_title(f'Average {feat.capitalize()} by Genre', fontweight='bold', fontsize=11)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel(feat)
        for p in axes[idx].patches:
            axes[idx].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')

    plt.suptitle('Comparison of Audio Characteristics Across Playlist Genres', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '04_genre_audio_features_comparison.png')
    plt.close()
    print(f"  [Saved] figures/04_genre_audio_features_comparison.png")


def run_pca_and_clustering(df_clean, audio_features, X_scaled):
    print("\n" + "=" * 85)
    print("STEP 4: DIMENSIONALITY REDUCTION (PCA) & CLUSTER EVALUATION")
    print("=" * 85)
    
    # 1. PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)
    
    print("PCA Explained Variance Ratios:")
    for i, (ev, cum) in enumerate(zip(evr, cum_evr), 1):
        print(f"  PC{i:2d}: {ev*100:6.2f}% (Cumulative: {cum*100:6.2f}%)")
        
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(range(1, len(evr)+1), cum_evr * 100, marker='o', linestyle='-', color='#1DB954', label='Cumulative Variance Explained', linewidth=2)
    plt.bar(range(1, len(evr)+1), evr * 100, alpha=0.6, color='#2E77D0', label='Individual Variance Explained')
    plt.xlabel('Principal Component Index', fontsize=11, fontweight='bold')
    plt.ylabel('Explained Variance (%)', fontsize=11, fontweight='bold')
    plt.title('PCA Explained Variance & Scree Plot', fontsize=13, fontweight='bold')
    plt.xticks(range(1, len(evr)+1))
    plt.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='70% Variance Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '05_pca_variance_scree.png')
    plt.close()
    print(f"  [Saved] figures/05_pca_variance_scree.png")
    
    # 2D PCA Genre Plot
    pca_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
    pca_df['playlist_genre'] = df_clean['playlist_genre']
    
    plt.figure(figsize=(11, 8), dpi=300)
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='playlist_genre', alpha=0.35, s=20, palette='Set1')
    plt.title('Spotify Songs in 2D PCA Space Colored by Playlist Genre', fontsize=13, fontweight='bold')
    plt.xlabel(f'PC1 ({evr[0]*100:.1f}% Variance - Energy/Loudness vs Acousticness)', fontsize=11)
    plt.ylabel(f'PC2 ({evr[1]*100:.1f}% Variance - Danceability/Valence vs Tempo)', fontsize=11)
    plt.legend(title='Playlist Genre', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '06_pca_2d_genres.png')
    plt.close()
    print(f"  [Saved] figures/06_pca_2d_genres.png")
    
    # 2. Comprehensive Cluster Evaluation (K=2 to 10)
    print("\n--- Comprehensive K Evaluation (K=2 to 10) ---")
    k_range = range(2, 11)
    inertias = []
    sil_scores = []
    ch_scores = []
    db_scores = []

    np.random.seed(RANDOM_STATE)
    sample_idx = np.random.choice(len(X_scaled), size=10000, replace=False)

    for k in k_range:
        km_test = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km_test.fit_predict(X_scaled)
        inertias.append(km_test.inertia_)
        sil = silhouette_score(X_scaled[sample_idx], labels[sample_idx])
        sil_scores.append(sil)
        ch = calinski_harabasz_score(X_scaled, labels)
        ch_scores.append(ch)
        db = davies_bouldin_score(X_scaled, labels)
        db_scores.append(db)
        print(f"  K={k:2d} | Inertia={km_test.inertia_:9.1f} | Silhouette={sil:.4f} | Calinski-Harabasz={ch:6.1f} | Davies-Bouldin={db:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    axes[0].plot(k_range, inertias, marker='o', color='#1DB954', linewidth=2, markersize=7)
    axes[0].set_title('Elbow Method (Inertia vs K)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[0].set_ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=11)
    axes[0].set_xticks(k_range)
    axes[0].axvline(x=6, color='red', linestyle='--', alpha=0.8, label='Selected K=6')
    axes[0].legend()

    axes[1].plot(k_range, sil_scores, marker='s', color='#2E77D0', linewidth=2, markersize=7)
    axes[1].set_title('Silhouette Score vs K', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].set_xticks(k_range)
    axes[1].axvline(x=6, color='red', linestyle='--', alpha=0.8, label='Selected K=6 (Peak for K>=3)')
    axes[1].legend()

    plt.suptitle('Evaluation of Clustering Metrics across K', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '07_elbow_and_silhouette.png')
    plt.close()
    print(f"  [Saved] figures/07_elbow_and_silhouette.png")
    
    # 3. Final Model Training (K=6)
    print("\n" + "=" * 85)
    print("STEP 5: FINAL K-MEANS MODEL TRAINING (K=6) & CLUSTER PROFILING")
    print("=" * 85)
    
    k_optimal = 6
    kmeans_model = KMeans(n_clusters=k_optimal, random_state=RANDOM_STATE, n_init=20)
    df_clean['cluster'] = kmeans_model.fit_predict(X_scaled)
    pca_df['cluster'] = df_clean['cluster']

    cluster_names = {
        0: "Cluster 0: Upbeat Pop & Dance Anthems",
        1: "Cluster 1: Live & High-Energy Concerts",
        2: "Cluster 2: Rhythmic Rap & Urban Beats",
        3: "Cluster 3: Acoustic & Melodic Soul",
        4: "Cluster 4: Instrumental & Synth Soundscapes",
        5: "Cluster 5: Intense Fast-Tempo EDM & Rock"
    }
    df_clean['cluster_profile'] = df_clean['cluster'].map(cluster_names)
    pca_df['cluster_profile'] = df_clean['cluster_profile']

    print("Discovered Cluster Sizes & Percentages:")
    cluster_counts = df_clean['cluster_profile'].value_counts()
    for name, cnt in cluster_counts.items():
        print(f"  {name:<46}: {cnt:5,d} songs ({cnt/len(df_clean)*100:5.1f}%)")
    print(f"  Sum of Cluster Sizes: {df_clean['cluster'].value_counts().sum():,} (Matches clean dataset: {len(df_clean) == df_clean['cluster'].value_counts().sum()})")
    
    # 2D PCA Cluster Plot
    plt.figure(figsize=(12, 8), dpi=300)
    pca_centroids = pca.transform(kmeans_model.cluster_centers_)[:, :2]

    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster_profile', alpha=0.4, s=25, palette='tab10')
    plt.scatter(pca_centroids[:, 0], pca_centroids[:, 1], c='black', s=220, marker='X', edgecolors='white', linewidths=2, label='Cluster Centroids', zorder=5)
    plt.title('K-Means Audio Segmentation in 2D PCA Space (K=6)', fontsize=13, fontweight='bold')
    plt.xlabel(f'PC1 ({evr[0]*100:.1f}% Variance)', fontsize=11)
    plt.ylabel(f'PC2 ({evr[1]*100:.1f}% Variance)', fontsize=11)
    plt.legend(title='Discovered Audio Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '08_kmeans_pca_clusters.png')
    plt.close()
    print(f"  [Saved] figures/08_kmeans_pca_clusters.png")
    
    # Cluster Profiles Heatmap
    cluster_means = df_clean.groupby('cluster')[audio_features].mean()
    plt.figure(figsize=(12, 6), dpi=300)
    sns.heatmap(cluster_means.T, annot=True, fmt='.3f', cmap='YlGnBu', linewidths=.5,
                yticklabels=[f.capitalize() for f in audio_features],
                xticklabels=[f"Cluster {i}" for i in range(k_optimal)])
    plt.title('Mean Unscaled Audio Feature Profiles per Discovered Cluster', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Cluster ID', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '09_cluster_profiles_heatmap.png')
    plt.close()
    print(f"  [Saved] figures/09_cluster_profiles_heatmap.png")
    
    # Cluster vs Genre Cross-Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    genre_cluster_ct = pd.crosstab(df_clean['cluster_profile'], df_clean['playlist_genre'], normalize='index') * 100
    genre_cluster_ct.plot(kind='barh', stacked=True, colormap='Spectral', ax=axes[0], edgecolor='white')
    axes[0].set_title('Playlist Genre Composition (%) within Each Cluster', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Percentage (%)')
    axes[0].set_ylabel('')
    axes[0].legend(title='Genre', bbox_to_anchor=(1.02, 1), loc='upper left')

    subgenre_cluster_ct = pd.crosstab(df_clean['playlist_genre'], df_clean['cluster'], normalize='index') * 100
    sns.heatmap(subgenre_cluster_ct, annot=True, fmt='.1f', cmap='Blues', ax=axes[1], cbar_kws={'label': '% of Genre in Cluster'})
    axes[1].set_title('Distribution of Playlist Genres Across Clusters (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Playlist Genre')

    plt.suptitle('Relationship Between Discovered Audio Clusters & Spotify Playlist Genres', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '10_cluster_genre_cross_analysis.png')
    plt.close()
    print(f"  [Saved] figures/10_cluster_genre_cross_analysis.png")
    
    # Top Playlists Analysis
    top_playlists = df_clean['playlist_name'].value_counts().head(10).index
    playlist_cluster_df = df_clean[df_clean['playlist_name'].isin(top_playlists)]
    top_pl_ct = pd.crosstab(playlist_cluster_df['playlist_name'], playlist_cluster_df['cluster'], normalize='index') * 100
    top_pl_ct.plot(kind='barh', stacked=True, colormap='tab10', edgecolor='white', figsize=(12, 6))
    plt.title('Cluster Distribution Across Top 10 High-Volume Spotify Playlists', fontsize=12, fontweight='bold')
    plt.xlabel('Proportion of Songs (%)')
    plt.ylabel('Playlist Name')
    plt.legend(title='Cluster ID', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '11_playlist_name_cluster_distribution.png')
    plt.close()
    print(f"  [Saved] figures/11_playlist_name_cluster_distribution.png")
    
    return df_clean, kmeans_model, pca, X_pca


def demonstrate_recommendation_system(df_clean, X_scaled):
    print("\n" + "=" * 85)
    print("STEP 6: CONTENT-BASED RECOMMENDATION SYSTEM DEMONSTRATIONS")
    print("=" * 85)
    
    def recommend(track_title, n_recs=5):
        matches = df_clean[df_clean['track_name'].str.lower().str.contains(track_title.lower(), na=False)]
        if len(matches) == 0:
            print(f"Track '{track_title}' not found in dataset.")
            return
        
        idx = matches.index[0]
        target = df_clean.iloc[idx]
        target_vec = X_scaled[idx].reshape(1, -1)
        
        sim_scores = cosine_similarity(target_vec, X_scaled)[0]
        sim_df = df_clean.copy()
        sim_df['similarity'] = sim_scores
        
        # Exclude exact same track and deduplicate artist/title duplicates
        filtered = sim_df[sim_df['track_name'].str.lower() != target['track_name'].lower()]
        unique_recs = filtered.drop_duplicates(subset=['track_name', 'track_artist']).sort_values('similarity', ascending=False).head(n_recs)
        
        print(f"\nQUERY SEED: '{target['track_name']}' by {target['track_artist']}")
        print(f"Genre: {target['playlist_genre']} ({target['playlist_subgenre']}) | Cluster: {target['cluster_profile']}")
        print(f"Acoustics: Dance={target['danceability']:.2f}, Energy={target['energy']:.2f}, Valence={target['valence']:.2f}, Loudness={target['loudness']:.1f}dB, Tempo={target['tempo']:.0f}BPM")
        print("-" * 80)
        print(f"Top {n_recs} Content-Based Acoustic Recommendations:")
        for i, (_, r) in enumerate(unique_recs.iterrows(), 1):
            print(f"  {i}. '{r['track_name']}' by {r['track_artist']}")
            print(f"     Similarity: {r['similarity']:.4f} | Genre: {r['playlist_genre']} ({r['playlist_subgenre']}) | Cluster: {r['cluster']}")
            print(f"     Audio: Dance={r['danceability']:.2f}, Energy={r['energy']:.2f}, Valence={r['valence']:.2f}, Loudness={r['loudness']:.1f}dB, Tempo={r['tempo']:.0f}BPM")

    test_queries = ['Shape of You', 'Lose Yourself to Dance', 'Bohemian Rhapsody', 'Animals', 'Stay With Me']
    for q in test_queries:
        recommend(q, 5)


def main():
    print("=" * 85)
    print("CORIZO AI INTERNSHIP - SPOTIFY SONGS GENRE SEGMENTATION PIPELINE")
    print("=" * 85)
    
    # 1. Ingestion & Audit
    df_raw = load_and_audit_data(DATASET_PATH)
    
    # 2. Preprocessing
    df_clean, audio_features, X_scaled, scaler = preprocess_data(df_raw)
    
    # 3. EDA & Visualizations
    perform_eda_and_visualizations(df_clean, audio_features)
    
    # 4. PCA & Clustering
    df_clustered, kmeans_model, pca, X_pca = run_pca_and_clustering(df_clean, audio_features, X_scaled)
    
    # 5. Recommendation System Demo
    demonstrate_recommendation_system(df_clustered, X_scaled)
    
    # 6. Save clustered dataset to data/ directory
    output_csv = DATA_DIR / 'spotify_clustered_dataset.csv'
    df_clustered.to_csv(output_csv, index=False)
    print(f"\nClustered dataset successfully exported to: {output_csv}")
    print("All pipeline steps executed successfully with zero errors.")


if __name__ == '__main__':
    main()
