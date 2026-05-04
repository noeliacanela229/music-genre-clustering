import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids

import warnings
warnings.filterwarnings('ignore')

# load dataset
url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
df = pd.read_csv(url)
print("dataset shape:", df.shape)
print("columns:", df.columns.tolist())

# look at the data
print("\nfirst 5 rows:")
print(df.head())
print("\ndata types:")
print(df.dtypes)
print("\nsummary stats:")
print(df.describe())
print("\ngenre counts:")
print(df['playlist_genre'].value_counts())

# clean up missing values and duplicates
print("\nmissing values:")
print(df.isnull().sum())
df = df.dropna()
df = df.drop_duplicates()
print("shape after cleaning:", df.shape)

# these are the audio features we want to use
# we drop things like track name and artist since those arent numbers
audio_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

genre_labels = df['playlist_genre'].reset_index(drop=True)
X = df[audio_features].reset_index(drop=True)
print("feature matrix shape:", X.shape)

# correlation matrix
plt.figure(figsize=(11, 8))
sns.heatmap(X.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5, square=True)
plt.title('Correlation Matrix of Audio Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("saved: correlation_matrix.png")

# scale the features
# this is important because tempo goes up to like 200 but danceability is only 0-1
# without scaling kmeans would just cluster on tempo basically
print("\nscaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("scaling complete.")

# feature distributions
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, feature in enumerate(audio_features):
    axes[i].hist(X[feature], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[i].set_title(feature, fontsize=12)
    axes[i].set_xlabel('value')
    axes[i].set_ylabel('count')
plt.suptitle('Audio Feature Distributions', fontsize=16)
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("saved: feature_distributions.png")

# elbow method to figure out the best k
print("\nrunning elbow method...")
inertia = []
k_range = range(2, 16)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    print(f"k={k} inertia={km.inertia_:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(k_range, inertia, marker='o', color='green', linewidth=2)
plt.title('Elbow Method — Optimal K for K-Means', fontsize=16)
plt.xlabel('number of clusters (k)')
plt.ylabel('inertia')
plt.xticks(k_range)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("saved: elbow_curve.png")

# k=8 based on elbow curve
k = 8

# algorithm 1: kmeans baseline
print("\nrunning kmeans...")
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_score = silhouette_score(X_scaled, kmeans_labels, sample_size=5000, random_state=42)
print("kmeans score:", round(kmeans_score, 4))

# algorithm 2: kmeans++ - smarter starting points than regular kmeans
print("\nrunning kmeans++...")
kmeanspp = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
kmeanspp_labels = kmeanspp.fit_predict(X_scaled)
kmeanspp_score = silhouette_score(X_scaled, kmeanspp_labels, sample_size=5000, random_state=42)
print("kmeans++ score:", round(kmeanspp_score, 4))

# algorithm 3: kmedoids - uses actual songs as centers instead of averages
print("\nrunning kmedoids...")
kmedoids = KMedoids(n_clusters=k, random_state=42)
kmedoids_labels = kmedoids.fit_predict(X_scaled)
kmedoids_score = silhouette_score(X_scaled, kmedoids_labels, sample_size=5000, random_state=42)
print("kmedoids score:", round(kmedoids_score, 4))

# algorithm 4: dbscan - groups by density, no need to pick k
# songs that dont fit anywhere get labeled as noise (-1)
print("\nrunning dbscan...")
dbscan = DBSCAN(eps=0.8, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print("dbscan clusters found:", n_clusters_dbscan)
print("noise points:", n_noise)
if n_clusters_dbscan > 1:
    mask = dbscan_labels != -1
    dbscan_score = silhouette_score(X_scaled[mask], dbscan_labels[mask], sample_size=5000, random_state=42)
    print("dbscan score (no noise):", round(dbscan_score, 4))
else:
    dbscan_score = 0
    print("not enough clusters for silhouette score")

# algorithm 5: gmm - gives probabilities instead of hard cluster assignments
print("\nrunning gmm...")
gmm = GaussianMixture(n_components=k, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
gmm_score = silhouette_score(X_scaled, gmm_labels, sample_size=5000, random_state=42)
print("gmm score:", round(gmm_score, 4))

# algorithm 6: hierarchical clustering - builds a tree of how genres relate
# using a sample because it gets slow on the full dataset
print("\nrunning hierarchical clustering...")
sample_idx = np.random.choice(len(X_scaled), 5000, replace=False)
X_sample = X_scaled[sample_idx]
hierarchical = AgglomerativeClustering(n_clusters=k)
hierarchical_labels = hierarchical.fit_predict(X_sample)
hierarchical_score = silhouette_score(X_sample, hierarchical_labels)
print("hierarchical score:", round(hierarchical_score, 4))

# compare all silhouette scores
algorithms = ['K-Means', 'K-Means++', 'K-Medoids', 'GMM', 'Hierarchical']
scores = [kmeans_score, kmeanspp_score, kmedoids_score, gmm_score, hierarchical_score]

plt.figure(figsize=(9, 5))
bars = plt.bar(algorithms, scores, color=['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c'])
plt.title('Silhouette Score Comparison Across Algorithms')
plt.ylabel('silhouette score')
plt.ylim(0, max(scores) + 0.05)
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{score:.4f}', ha='center', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('silhouette_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("saved: silhouette_comparison.png")

# pca to visualize clusters in 2d
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("pca variance explained:", f"{sum(pca.explained_variance_ratio_):.2%}")

plt.figure(figsize=(11, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.4, s=8)
plt.colorbar(scatter, label='cluster')
plt.title(f'K-Means Clusters via PCA (K={k})')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('pca_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("saved: pca_plot.png")

# heatmap showing avg audio features per cluster
df_results = X.copy()
df_results['cluster'] = kmeans_labels
df_results['genre'] = genre_labels.values

cluster_avg = df_results.groupby('cluster')[audio_features].mean().round(3)

plt.figure(figsize=(13, 6))
sns.heatmap(cluster_avg.T, annot=True, fmt='.2f', cmap='YlGn', linewidths=0.4)
plt.title('Average Audio Features per Cluster (K-Means)')
plt.ylabel('feature')
plt.xlabel('cluster')
plt.tight_layout()
plt.savefig('cluster_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("saved: cluster_heatmap.png")

# bar chart showing which genres ended up in each cluster
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i in range(k):
    top = df_results[df_results['cluster'] == i]['genre'].value_counts().head(5)
    axes[i].bar(top.index, top.values, color='steelblue')
    axes[i].set_title(f'cluster {i}')
    axes[i].set_ylabel('count')
    axes[i].tick_params(axis='x', rotation=30)
plt.suptitle('Top Genres per Cluster (K-Means)', fontsize=14)
plt.tight_layout()
plt.savefig('genre_bar_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print("saved: genre_bar_chart.png")

# print top genres per cluster
print("\ntop genres per cluster:")
for i in range(k):
    top = df_results[df_results['cluster'] == i]['genre'].value_counts().head(5)
    print(f"\ncluster {i}:")
    print(top.to_string())

# summary
print("\n--- summary ---")
print(f"dataset: {df.shape[0]:,} tracks, {df['playlist_genre'].nunique()} genres")
print(f"k: {k}")
print(f"kmeans:       {kmeans_score:.4f}")
print(f"kmeans++:     {kmeanspp_score:.4f}")
print(f"kmedoids:     {kmedoids_score:.4f}")
print(f"gmm:          {gmm_score:.4f}")
print(f"hierarchical: {hierarchical_score:.4f}")
print(f"dbscan clusters: {n_clusters_dbscan}, noise: {n_noise}")
print(f"pca variance: {sum(pca.explained_variance_ratio_):.2%}")