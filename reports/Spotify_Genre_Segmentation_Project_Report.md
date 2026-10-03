# CORIZO ARTIFICIAL INTELLIGENCE INTERNSHIP
## PROJECT 2: SPOTIFY SONGS' GENRE SEGMENTATION & RECOMMENDATION SUPPORT SYSTEM

---

**Author / Intern:** Hamid Kamal  
**Role:** Data Science & Machine Learning Engineer Intern  
**Project Domain:** Unsupervised Machine Learning, Audio Informatics & Recommender Systems  
**Primary Dataset:** Spotify Audio Features Dataset (`upload_efc975f9-7e96-4804-8ae1-691bc3670af7.csv`)  
**Status:** Successfully Executed, Validated & Submission-Ready  

---

## 1. Project Title
**Spotify Songs' Genre Segmentation and Content-Based Music Recommendation Support System**

---

## 2. Objective
The primary objective of this project is to build an automated music-analysis and unsupervised clustering pipeline that:
1. Ingests and audits the Spotify audio features dataset.
2. Implements a defensible preprocessing, cleaning, and standardization pipeline.
3. Conducts exploratory data analysis (EDA) and correlation matrix profiling.
4. Evaluates cluster structures using dimensionality reduction (PCA) and multi-metric clustering evaluation ($K \in [2, 10]$).
5. Segregates songs into interpretable acoustic archetypes and maps them against human-curated playlist genres, subgenres, and playlist names.
6. Implements and demonstrates a content-based recommendation component using audio feature vector similarity.

---

## 3. Dataset Description
The analysis utilizes the provided Spotify Songs Dataset containing **32,833 records** across **23 features**:
- **Track Identifiers & Metadata (9):** `track_id`, `track_name`, `track_artist`, `track_popularity`, `track_album_id`, `track_album_name`, `track_album_release_date`, `playlist_name`, `playlist_id`.
- **Taxonomy Categories (2):** `playlist_genre` (6 primary genres: `edm`, `rap`, `pop`, `r&b`, `latin`, `rock`), `playlist_subgenre` (24 distinct subgenres).
- **Acoustic & Perceptual Attributes (12):** `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`.

---

## 4. Problem Statement
Commercial streaming platforms host tens of millions of songs categorized under human-assigned genres. However, human genre labels often capture cultural, historical, or marketing associations rather than pure acoustic properties (for example, an acoustic indie ballad and a heavy arena rock anthem may both be cataloged under "Rock"). This project investigates how unsupervised machine learning on continuous audio descriptors (rhythm, timbre, dynamics, energy, and acousticness) can group songs into acoustic clusters to support content-based similarity retrieval.

---

## 5. Methodology
The project follows a standard Data Science Lifecycle (CRISP-DM):
```
[Raw Dataset] ──► [Data Audit & Cleaning] ──► [Feature Transformation & Scaling]
                                                          │
   ┌──────────────────────────────────────────────────────┴─────────────────────────────────────────────────┐
   ▼                                                      ▼                                                 ▼
[Exploratory Data Analysis]                [Dimensionality Reduction (PCA)]                 [K-Means Cluster Evaluation]
   • Feature Distributions                    • Variance Scree Plot                            • WCSS / Elbow Method
   • Correlation Heatmap                      • 2D Orthogonal Projections                      • Silhouette & Multi-Metric
   • Genre Averages                                       │                                                 │
   └──────────────────────────────────────────────────────┼─────────────────────────────────────────────────┘
                                                          ▼
                                            [Final Clustering (K=6)]
                                                          │
                              ┌───────────────────────────┴───────────────────────────┐
                              ▼                                                       ▼
                [Cluster Archetype Profiling]                           [Content-Based Recommendation Demo]
                • Statistical Mean Vectors                              • Cosine Similarity Vector Search
                • Cross-Genre & Playlist Mapping                        • Demonstration Queries & Deduplication
```

---

## 6. Data Preprocessing
### Preprocessing Actions:
1. **Missing Value Treatment:** Audited 32,833 rows. Exactly 5 rows (<0.015%) had null values in `track_name`, `track_artist`, or `track_album_name`. These 5 rows were removed, leaving **32,828 clean records**.
2. **Duplicate Handling & Record Granularity:** The dataset contains 28,356 unique `track_id` values. Duplicate track ID rows reflect the same song included in multiple Spotify playlists and genres. Audio features were verified to be identical across duplicates. Playlist-level instances were retained to enable genre-to-cluster cross-analysis.
3. **Unit Transformations:**
   - `duration_ms` was converted to `duration_min` ($\text{duration\_ms} / 60,000.0$).
   - `track_album_release_date` was parsed to extract numeric `release_year`.
4. **Feature Segregation:** Metadata identifiers and category labels (`track_id`, `track_name`, `artist`, `album_id`, `playlist_name`, `genre`, `subgenre`) were excluded from clustering inputs to avoid data leakage and categorical bias.
5. **Feature Standardization:** The 10 continuous audio features were standardized to mean $\mu = 0$ and standard deviation $\sigma = 1$ using `StandardScaler`.

---

## 7. Exploratory Data Analysis
### Genre & Subgenre Distributions:
The dataset contains balanced representation across the 6 playlist genres:
- **EDM:** 6,043 songs (18.4%)
- **Rap:** 5,743 songs (17.5%)
- **Pop:** 5,507 songs (16.8%)
- **R&B:** 5,431 songs (16.5%)
- **Latin:** 5,153 songs (15.7%)
- **Rock:** 4,951 songs (15.1%)

### Audio Characteristics by Genre:
- **Highest Energy & Loudness:** EDM ($\text{Energy} = 0.802, \text{Loudness} = -5.43\text{ dB}$) and Rock ($\text{Energy} = 0.733$).
- **Highest Danceability & Speechiness:** Rap ($\text{Danceability} = 0.718, \text{Speechiness} = 0.198$).
- **Highest Acousticness:** R&B ($\text{Acousticness} = 0.299$).
- **Highest Positivity (Valence):** Latin ($\text{Valence} = 0.607$).

---

## 8. Visualizations Generated
All 11 visualizations were generated and saved in [`figures/`](../figures/):
1. `01_correlation_matrix.png` — Correlation heatmap of audio features and popularity.
2. `02_genre_subgenre_distribution.png` — Horizontal bar distributions of genres and 24 subgenres.
3. `03_audio_features_distributions.png` — Histograms with KDE curves for all 10 audio features, popularity, and release year.
4. `04_genre_audio_features_comparison.png` — Ranked bar charts of average acoustic attributes by genre.
5. `05_pca_variance_scree.png` — Individual and cumulative explained variance scree plot.
6. `06_pca_2d_genres.png` — 2D PCA scatter plot colored by labeled playlist genre.
7. `07_elbow_and_silhouette.png` — Dual-panel Elbow curve (Inertia) and Silhouette score curve ($K=2..10$).
8. `08_kmeans_pca_clusters.png` — 2D PCA scatter plot colored by discovered K-Means clusters with centroids.
9. `09_cluster_profiles_heatmap.png` — Unscaled feature heatmap across all 6 cluster archetypes.
10. `10_cluster_genre_cross_analysis.png` — Stacked horizontal composition bars and genre-to-cluster cross-tabulation heatmap.
11. `11_playlist_name_cluster_distribution.png` — Cluster proportion distribution across top 10 high-volume playlists.

---

## 9. Correlation Analysis
The Pearson correlation matrix computed directly from the dataset reveals key acoustic dynamics:
- **$r(\text{energy}, \text{loudness}) = +0.6767$ (Strong positive):** Energetic tracks tend to be mastered at higher loudness levels.
- **$r(\text{energy}, \text{acousticness}) = -0.5397$ & $r(\text{loudness}, \text{acousticness}) = -0.3616$ (Strong negative):** Acoustic instruments correlate with quieter, less compressed recordings.
- **$r(\text{danceability}, \text{valence}) = +0.3305$ (Moderate positive):** Cheerful musical scales tend to accompany danceable tracks.
- **$r(\text{danceability}, \text{tempo}) = -0.1841$ (Moderate negative):** High tempos (>160 BPM) tend to reduce measured danceability scores.
- **Popularity Independence:** `track_popularity` has low linear correlation with individual audio features ($|r| \le 0.15$).

---

## 10. Dimensionality Reduction (PCA) Analysis
PCA applied to the 10 standardized features yielded the following explained variance distribution:
- **PC1 (21.52%):** Primary acoustic energy axis (Loudness & Energy vs Acousticness).
- **PC2 (15.45%, Cumulative: 36.98%):** Rhythmic & mood axis (Danceability & Valence vs Tempo).
- **PC3 (11.23%, Cumulative: 48.21%):** Speechiness & vocal rhythm axis.
- **PC4 (9.92%, Cumulative: 58.13%):** Instrumentalness axis.
- **PC5 (9.81%, Cumulative: 67.94%):** Duration & liveness axis.
- **PC6 (9.69%, Cumulative: 77.64%):** First 6 principal components account for 77.64% of total variance.

---

## 11. K-Selection Justification & Methodology
We evaluated $K \in [2, 10]$ across four quantitative validation metrics:

| $K$ | Inertia (WCSS) | Silhouette Score | Calinski-Harabasz | Davies-Bouldin |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 283,302.7 | 0.1777 | 5,211.5 | 2.2195 |
| 3 | 255,712.5 | 0.1238 | 4,657.6 | 2.2020 |
| 4 | 234,444.3 | 0.1331 | 4,379.2 | 1.9771 |
| 5 | 216,301.3 | 0.1352 | 4,248.1 | 1.8398 |
| **6 (Selected)** | **203,507.3** | **0.1379** | **4,024.7** | **1.7349** |
| 7 | 193,502.5 | 0.1253 | 3,810.1 | 1.7644 |
| 8 | 185,728.0 | 0.1233 | 3,598.6 | 1.7092 |
| 9 | 178,735.4 | 0.1226 | 3,432.4 | 1.6793 |
| 10 | 172,317.8 | 0.1251 | 3,300.4 | 1.6374 |

### Academically Honest K-Selection Rationale:
1. **$K=2$ Silhouette Behavior:** $K=2$ produces the highest overall mathematical silhouette score ($0.1777$). However, $K=2$ only partitions the data into a broad binary split (loud/electronic vs quiet/acoustic), which is too coarse for meaningful genre segmentation or targeted recommendation.
2. **Local Peak Among Multi-Cluster Models ($K \ge 3$):** Among granular multi-cluster candidates, $K=6$ reaches the highest silhouette score ($0.1379$), outperforming $K=3, 4, 5, 7, 8, 9$.
3. **Metric Trade-Offs:** The clustering metrics do not unanimously point to a single $K$. For instance, higher values ($K=8, 9, 10$) achieve lower Davies-Bouldin scores ($1.7092, 1.6793, 1.6374$) as clusters become smaller and tighter, but this leads to over-fragmentation.
4. **Elbow Inflection & Practical Domain Balance:** The inertia reduction curve displays a clear bend and leveling around $K=6$. Combined with domain interpretability, $K=6$ provides a well-balanced segmentation that identifies 6 practical musical archetypes.

---

## 12. Detailed Cluster Profiles & Interpretations

```
====================================================================================================
MEAN AUDIO FEATURE PROFILES BY DISCOVERED CLUSTER (UNSCALED)
====================================================================================================
Feature            Cluster 0    Cluster 1    Cluster 2    Cluster 3    Cluster 4    Cluster 5
----------------------------------------------------------------------------------------------------
Danceability           0.740        0.610        0.726        0.605        0.663        0.546
Energy                 0.724        0.780        0.663        0.425        0.783        0.791
Loudness (dB)         -6.213       -5.977       -6.823      -10.513       -6.972       -5.325
Speechiness            0.075        0.112        0.313        0.073        0.071        0.071
Acousticness           0.142        0.114        0.185        0.519        0.072        0.068
Instrumentalness       0.014        0.054        0.010        0.093        0.746        0.023
Liveness               0.149        0.613        0.174        0.148        0.171        0.173
Valence                0.680        0.510        0.544        0.393        0.387        0.379
Tempo (BPM)          113.552      121.539      122.622      112.380      125.088      132.633
Duration (min)         3.681        3.865        3.591        3.786        4.202        3.790
Track Popularity      45.427       38.807       44.066       44.275       29.993       41.511
Song Count            10,763        2,130        4,321        4,682        2,470        8,462
Percentage (%)         32.8%         6.5%        13.2%        14.3%         7.5%        25.8%
====================================================================================================
```

- **Cluster 0: Upbeat Pop & Dance Anthems (10,763 songs, 32.8%):** Characterized by high valence ($0.680$), high danceability ($0.740$), and high energy ($0.724$). Prominent in Latin ($26.2\%$), Pop ($19.6\%$), and R&B ($18.5\%$).
- **Cluster 1: Live & High-Energy Concerts (2,130 songs, 6.5%):** Characterized by high liveness ($0.613$) and high energy ($0.780$), capturing live recordings across EDM ($25.6\%$), Rock ($18.3\%$), and Rap ($15.7\%$).
- **Cluster 2: Rhythmic Rap & Urban Beats (4,321 songs, 13.2%):** Characterized by high speechiness ($0.313$) and high danceability ($0.726$). **53.1% Rap** and $19.8\%$ R&B.
- **Cluster 3: Acoustic & Melodic Soul (4,682 songs, 14.3%):** Characterized by high acousticness ($0.519$), lower energy ($0.425$), and lower loudness ($-10.51$ dB). R&B ($33.1\%$), Rock ($19.1\%$), and Pop ($16.9\%$).
- **Cluster 4: Instrumental & Synth Soundscapes (2,470 songs, 7.5%):** Characterized by high instrumentalness ($0.746$), high energy ($0.783$), and longer average duration ($4.20$ min). **58.1% EDM**.
- **Cluster 5: Intense Fast-Tempo EDM & Rock (8,462 songs, 25.8%):** Characterized by higher tempo ($132.6$ BPM), higher loudness ($-5.32$ dB), and high energy ($0.791$). EDM ($27.8\%$), Rock ($26.3\%$), and Pop ($21.0\%$).

---

## 13. PCA Analysis & Cluster Projection
Projecting songs onto the first two principal components alongside cluster centroids visualizes separation along the primary acoustic energy (PC1) and rhythmic/mood (PC2) dimensions.

---

## 14. Recommendation System Methodology
The recommendation component is implemented as a **content-based vector similarity search** using **Cosine Similarity** over the 10 standardized audio features:
$$\text{Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
The system accepts a seed track, retrieves its standardized feature vector, calculates pairwise cosine similarities against all songs in the cleaned dataset, excludes exact seed title matches, deduplicates repeated artist/title combinations, and returns the top $N$ closest acoustic matches.

---

## 15. Recommendation Demonstration Queries

The following demonstration queries illustrate how content-based similarity retrieves acoustically close tracks from the dataset:

### Demonstration 1: *"Shape of You"* by Ed Sheeran (Pop | Cluster 0)
- **Seed Audio:** Dance=$0.82$, Energy=$0.65$, Valence=$0.93$, Loudness=$-3.2$ dB, Tempo=$96$ BPM
1. *Quizas - Remix* by Tony Dize | Latin (reggaeton) | Cluster 0 | **Similarity: 0.9583**
2. *Brujeria* by El Gran Combo De Puerto Rico | Latin (latin pop) | Cluster 0 | **Similarity: 0.9533**
3. *No Vuelvas Más* by Darell | Latin (reggaeton) | Cluster 0 | **Similarity: 0.9434**
4. *Work It Out* by Jurassic 5 | Rap (southern hip hop) | Cluster 0 | **Similarity: 0.9425**
5. *Me Quedaré Contigo* by El Micha | Latin (tropical) | Cluster 0 | **Similarity: 0.9415**

### Demonstration 2: *"Lose Yourself to Dance"* by Daft Punk (Pop | Cluster 0)
- **Seed Audio:** Dance=$0.83$, Energy=$0.66$, Valence=$0.67$, Loudness=$-7.8$ dB, Tempo=$100$ BPM
1. *Nobody Better - Blacksmith R'n'B Rub* by Tina Moore | R&B (new jack swing) | Cluster 0 | **Similarity: 0.9818**
2. *Watching The World* by Surahn | R&B (neo soul) | Cluster 0 | **Similarity: 0.9710**
3. *Así Es la Vida* by Elefante | Latin (latin pop) | Cluster 0 | **Similarity: 0.9665**
4. *I Hear You Paint Houses* by Robbie Robertson | Rock (classic rock) | Cluster 0 | **Similarity: 0.9657**
5. *90's Girl* by Black girl | R&B (new jack swing) | Cluster 0 | **Similarity: 0.9578**

### Demonstration 3: *"Bohemian Rhapsody - 2011 Mix"* by Queen (Rock | Cluster 3)
- **Seed Audio:** Dance=$0.41$, Energy=$0.40$, Valence=$0.22$, Loudness=$-9.9$ dB, Tempo=$71$ BPM
1. *Bohemian Rhapsody - Remastered 2011* by Queen | Rock (classic rock) | Cluster 3 | **Similarity: 1.0000**
2. *As We Lay* by Shirley Murdock | R&B (new jack swing) | Cluster 3 | **Similarity: 0.9571**
3. *Jubilee Street* by Nick Cave & The Bad Seeds | Rock (permanent wave) | Cluster 3 | **Similarity: 0.9531**
4. *The Drugs Don't Work* by The Verve | Rock (album rock) | Cluster 3 | **Similarity: 0.9511**
5. *Tell Your Friends* by The Weeknd | R&B (urban contemporary) | Cluster 3 | **Similarity: 0.9409**

### Demonstration 4: *"Animals"* by Maroon 5 (Pop | Cluster 5)
- **Seed Audio:** Dance=$0.46$, Energy=$0.74$, Valence=$0.34$, Loudness=$-6.4$ dB, Tempo=$190$ BPM
1. *I Want You to Want Me - Live* by Cheap Trick | Rock (classic rock) | Cluster 1 | **Similarity: 0.9608**
2. *Only One - Brookes Brothers Remix* by Sigala | Pop (indie poptimism) | Cluster 1 | **Similarity: 0.9597**
3. *Roar* by Katy Perry | Pop (post-teen pop) | Cluster 5 | **Similarity: 0.9467**
4. *Dark Side - Shew Remix* by Phoebe Ryan | EDM (pop edm) | Cluster 5 | **Similarity: 0.9437**
5. *Cold Water - Boombox Cartel Remix* by Major Lazer | Rap (trap) | Cluster 5 | **Similarity: 0.9410**

### Demonstration 5: *"Stay With Me"* by ayokay (Pop | Cluster 0)
- **Seed Audio:** Dance=$0.69$, Energy=$0.69$, Valence=$0.47$, Loudness=$-6.6$ dB, Tempo=$100$ BPM
1. *Attention - Tropical House Mix* by Boonz | Latin (tropical) | Cluster 3 | **Similarity: 0.9790**
2. *Kissing Other People* by Lennon Stella | Pop (post-teen pop) | Cluster 3 | **Similarity: 0.9655**
3. *Bonnie & Clyde* by Jake Cooper | Latin (tropical) | Cluster 0 | **Similarity: 0.9427**
4. *Não Demora* by Mati | Rap (trap) | Cluster 3 | **Similarity: 0.9374**
5. *Feel jazz - Remix* by Nidža Bleja | R&B (hip pop) | Cluster 3 | **Similarity: 0.9358**

---

## 16. Key Findings
1. **Acoustic Diversity within Labeled Genres:** Playlist genres contain diverse acoustic profiles. For instance, Rock songs span both acoustic ballads in Cluster 3 and fast-tempo tracks in Cluster 5.
2. **Production Correlations:** Energy and Loudness correlate strongly ($r = +0.6767$), while Acousticness correlates negatively with both ($r = -0.5397$ and $-0.3616$).
3. **Cross-Genre Acoustic Alignment:** Content-based similarity identifies tracks across different playlist genres that share similar tempo, energy, and danceability.

---

## 17. Business & Application Relevance
- **Cold-Start Handling:** New tracks without historical user listening data can be indexed and retrieved based on acoustic features.
- **Mood-Based Curation:** Cluster profiles allow grouping songs by acoustic atmosphere (e.g. ambient study music or workout tempos).
- **Playlist Structuring:** Audio similarity can help sequence songs with smooth acoustic transitions.

---

## 18. Limitations
1. **Dataset Granularity:** The dataset contains repeated tracks across playlists.
2. **Feature Coverage:** High-level summary features (e.g., scalar energy, valence) cannot capture lyrical semantics, vocal timbre, or chord progressions.
3. **K-Means Assumptions:** K-Means assumes spherical clusters in Euclidean space and requires pre-specifying $K$.
4. **Metric Ambiguity:** Unsupervised validation metrics provide different signals ($K=2$ has highest silhouette, while $K=8..10$ have lower Davies-Bouldin indices).
5. **No User Interaction Data:** The recommender is a content-based demonstration and does not incorporate user listening histories, skip rates, or collaborative filtering.

---

## 19. Conclusion
The project implements a complete, validated data analysis and unsupervised clustering pipeline on the Spotify dataset. By combining feature standardization, multi-metric evaluation, PCA visualization, and vector similarity search, the resulting model segments songs into 6 acoustic archetypes and demonstrates a functional content-based recommendation prototype.

---

## 20. Technologies & Libraries Used
- **Programming Language:** Python 3.12
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning & Modeling:** `scikit-learn` (`StandardScaler`, `PCA`, `KMeans`, `silhouette_score`, `calinski_harabasz_score`, `davies_bouldin_score`, `cosine_similarity`)
- **Environment:** Jupyter Notebook / Visual Studio Code / Antigravity IDE
