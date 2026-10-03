# Data Directory: Spotify Songs' Genre Segmentation

This directory contains the dataset documentation, the raw dataset location guidance, and the preprocessed clustered output dataset.

---

## 1. Raw Dataset Information

* **Dataset Name:** Spotify Audio Features Dataset
* **Raw Records:** 32,833 songs across 23 attributes
* **Expected Raw Filename:** `upload_efc975f9-7e96-4804-8ae1-691bc3670af7.csv` (or `data/upload_efc975f9-7e96-4804-8ae1-691bc3670af7.csv`)
* **Primary Source:** Spotify Web API (Echo Nest audio feature extraction)
* **Metadata Fields (9):** `track_id`, `track_name`, `track_artist`, `track_popularity`, `track_album_id`, `track_album_name`, `track_album_release_date`, `playlist_name`, `playlist_id`.
* **Taxonomy Fields (2):** `playlist_genre` (6 genres: `edm`, `rap`, `pop`, `r&b`, `latin`, `rock`), `playlist_subgenre` (24 subgenres).
* **Audio Descriptors (12):** `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`.

---

## 2. Processed & Clustered Dataset

* **File:** `spotify_clustered_dataset.csv`
* **Clean Records:** 32,828 songs (5 records with missing track metadata removed, <0.015%)
* **Added Features:**
  - `duration_min`: Track duration converted from milliseconds to minutes.
  - `release_year`: Extracted numeric album release year.
  - `cluster`: Numerical cluster assignment ($0 \dots 5$).
  - `cluster_profile`: Descriptive cluster archetype label.

---

## 3. How to Run Locally

Place the raw CSV file in either the repository root or the `data/` directory:
```bash
# Recommended placement:
data/upload_efc975f9-7e96-4804-8ae1-691bc3670af7.csv
```
The automated script [`src/spotify_genre_segmentation.py`](../src/spotify_genre_segmentation.py) and the notebook [`notebooks/Spotify_Genre_Segmentation.ipynb`](../notebooks/Spotify_Genre_Segmentation.ipynb) automatically search both locations.
