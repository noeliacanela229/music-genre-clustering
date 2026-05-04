# Clustering Music Genres
Noelia Canela & Ethan Ferguson — CS 472 Final Project - Music Clustering

## What this is
We wanted to see if a clustering algorithm could figure out music genres on its own just from audio data — no labels, no hints. We used a Spotify dataset and ran 6 different clustering algorithms to compare which one works best.

## Dataset
We used the TidyTuesday Spotify dataset which has about 32,000 songs across 6 genres (edm, rap, pop, r&b, latin, rock). It loads straight from a URL in the script so you don't need to download anything manually.

The features we clustered on: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo

## Algorithms we ran
- K-Means
- K-Means++
- K-Medoids
- DBSCAN
- GMM (Gaussian Mixture Model)
- Hierarchical Clustering

## How to run it

Install the dependencies:
```
pip install -r requirements.txt
```

Run the script:
```
python3 music_genre_clustering.py
```

It'll download the dataset automatically and save all the plots to your folder. Close each plot window as it pops up so it keeps running.

## What it generates
- correlation_matrix.png
- feature_distributions.png
- elbow_curve.png
- silhouette_comparison.png
- pca_plot.png
- cluster_heatmap.png
- genre_bar_chart.png

## Results
K-Means and K-Means++ tied for the best silhouette score at 0.1384. DBSCAN really struggled — it found 46 clusters and labeled over half the songs as noise, which makes sense because music genres blend into each other a lot and aren't dense enough for density-based clustering to work well. GMM went negative which was surprising.

K=8 was picked based on the elbow curve.