# FINAL QA AND VERIFICATION REPORT
## Corizo Artificial Intelligence Internship — Project 2: Spotify Songs' Genre Segmentation

**Date & Time of QA Execution:** 2026-10-03  
**Auditor / Engineer:** Hamid Kamal  
**Workspace:** `d:\Spotify Genre Segmentation`  
**Dataset Source:** `upload_efc975f9-7e96-4804-8ae1-691bc3670af7.csv`  

---

### 1. Execution Status
* **Script executed successfully (`src/spotify_genre_segmentation.py`):** **YES** (Exit code: 0, zero errors, zero warnings)
* **Notebook generated & verified (`notebooks/Spotify_Genre_Segmentation.ipynb`):** **YES** (Valid nbformat v4 JSON, all sections executable)
* **Clustered dataset exported (`data/spotify_clustered_dataset.csv`):** **YES** (32,828 rows × 26 columns)
* **Figures generated in `figures/`:** **YES** (11/11 high-resolution figures verified)

---

### 2. Dataset Verification
* **Original rows:** 32,833
* **Cleaned rows:** 32,828 (5 null metadata rows dropped, <0.015%)
* **Total columns:** 23
* **Missing values in cleaned dataset:** 0
* **Unique track IDs:** 28,356
* **Playlist genres (6):** `edm` (6,043), `rap` (5,743), `pop` (5,507), `r&b` (5,431), `latin` (5,153), `rock` (4,951)
* **Playlist subgenres:** 24
* **Clustering features used (10):** `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_min`

---

### 3. Model Verification
* **Tested K values:** $K = 2, 3, 4, 5, 6, 7, 8, 9, 10$
* **Evaluation Metrics Table:**
  - $K=2$: Inertia=283,302.7 | **Silhouette=0.1777 (Highest overall)** | Calinski-Harabasz=5,211.5 | Davies-Bouldin=2.2195
  - $K=3$: Inertia=255,712.5 | Silhouette=0.1238 | Calinski-Harabasz=4,657.6 | Davies-Bouldin=2.2020
  - $K=4$: Inertia=234,444.3 | Silhouette=0.1331 | Calinski-Harabasz=4,379.2 | Davies-Bouldin=1.9771
  - $K=5$: Inertia=216,301.3 | Silhouette=0.1352 | Calinski-Harabasz=4,248.1 | Davies-Bouldin=1.8398
  - **$K=6$ (Selected):** Inertia=203,507.3 | **Silhouette=0.1379 (Highest for $K \ge 3$)** | Calinski-Harabasz=4,024.7 | Davies-Bouldin=1.7349
  - $K=7$: Inertia=193,502.5 | Silhouette=0.1253 | Calinski-Harabasz=3,810.1 | Davies-Bouldin=1.7644
  - $K=8$: Inertia=185,728.0 | Silhouette=0.1233 | Calinski-Harabasz=3,598.6 | Davies-Bouldin=1.7092
  - $K=9$: Inertia=178,735.4 | Silhouette=0.1226 | Calinski-Harabasz=3,432.4 | Davies-Bouldin=1.6793
  - $K=10$: Inertia=172,317.8 | Silhouette=0.1251 | Calinski-Harabasz=3,300.4 | Davies-Bouldin=1.6374
* **Selected K:** **6**
* **Reason / Justification:** 
  1. $K=2$ achieves the highest overall silhouette score ($0.1777$), but produces an uninformative binary macro-split (loud electronic vs quiet acoustic).
  2. $K=6$ reaches the highest silhouette score among multi-cluster models ($K \ge 3$) at $0.1379$.
  3. While $K=8..10$ achieve lower Davies-Bouldin scores ($1.7092..1.6374$), they lead to smaller, over-fragmented clusters.
  4. $K=6$ sits at the elbow leveling zone and balances quantitative performance with practical musical interpretability.

---

### 4. Cluster Verification
* **Cluster Sizes & Breakdown ($K=6$):**
  - Cluster 0 (*Upbeat Pop & Dance Anthems*): 10,763 songs (32.8%)
  - Cluster 1 (*Live & High-Energy Concerts*): 2,130 songs (6.5%)
  - Cluster 2 (*Rhythmic Rap & Urban Beats*): 4,321 songs (13.2%)
  - Cluster 3 (*Acoustic & Melodic Soul*): 4,682 songs (14.3%)
  - Cluster 4 (*Instrumental & Synth Soundscapes*): 2,470 songs (7.5%)
  - Cluster 5 (*Intense Fast-Tempo EDM & Rock*): 8,462 songs (25.8%)
* **Sum of cluster sizes:** $10,763 + 2,130 + 4,321 + 4,682 + 2,470 + 8,462 = 32,828$
* **Matches cleaned dataset:** **YES** ($32,828 == 32,828$)

---

### 5. Recommendation Verification
* **Demonstration Queries Executed:** 
  1. *Shape of You* by Ed Sheeran (Pop | Cluster 0) ──► Returned 5 similar tracks (e.g. *Quizas - Remix*, *Brujeria*, *No Vuelvas Más*)
  2. *Lose Yourself to Dance* by Daft Punk (Pop | Cluster 0) ──► Returned 5 similar tracks (e.g. *Nobody Better*, *Watching The World*)
  3. *Bohemian Rhapsody - 2011 Mix* by Queen (Rock | Cluster 3) ──► Returned 5 similar tracks (e.g. *As We Lay*, *Jubilee Street*, *The Drugs Don't Work*)
  4. *Animals* by Maroon 5 (Pop | Cluster 5) ──► Returned 5 similar tracks (e.g. *I Want You to Want Me - Live*, *Roar*, *Dark Side*)
  5. *Stay With Me* by ayokay (Pop | Cluster 0) ──► Returned 5 similar tracks (e.g. *Attention - Tropical House Mix*, *Kissing Other People*)
* **Recommendations generated successfully:** **YES** (All similarity scores $>0.93$, valid non-null floats, exact duplicate tracks excluded, seed omitted).

---

### 6. Report & Artifact Consistency Verification
* **All numbers verified across CSV ──► Script ──► Notebook ──► Clustered CSV ──► Report:** **YES**
* **All required Corizo project tasks completed:** **YES**
  - Preprocessing: ✅
  - EDA & Visualizations: ✅
  - Correlation Matrix: ✅
  - Cluster parameters (Genres, Subgenres, Playlist Names): ✅
  - Recommendation Demonstration: ✅

---

### 7. Remaining Issues
**No known issues remain.**

---
*End of Final QA Report.*
