# imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
import warnings
warnings.filterwarnings('ignore')

url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
df = pd.read_csv(url)
print("shape:", df.shape)
print(df.head())
print(df.dtypes)
print(df.describe())
print(df['playlist_genre'].value_counts())

print(df.isnull().sum())
df = df.dropna()
df = df.drop_duplicates()
print("after cleaning:", df.shape)

features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

genres = df['playlist_genre'].reset_index(drop=True)
X = df[features].reset_index(drop=True)

le = LabelEncoder()
le.fit_transform(genres)

# colors for each genre so we can see them in the plots
colors = {
    'edm': '#e74c3c',
    'latin': '#f39c12',
    'pop': '#2ecc71',
    'r&b': '#3498db',
    'rap': '#9b59b6',
    'rock': '#1abc9c'
}
point_colors = [colors[g] for g in genres]
patches = [mpatches.Patch(color=c, label=g) for g, c in colors.items()]

# correlation matrix to see how features relate before we do anything
plt.figure(figsize=(11, 8))
sns.heatmap(X.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5, square=True)
plt.title('Correlation Matrix of Audio Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# scale everything - important because tempo is 0-200 but danceability is 0-1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# distributions of each feature
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, f in enumerate(features):
    axes[i].hist(X[f], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[i].set_title(f)
    axes[i].set_xlabel('value')
    axes[i].set_ylabel('count')
plt.suptitle('Audio Feature Distributions', fontsize=14)
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# elbow method to pick k
inertia = []
for k in range(2, 16):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    print(f"k={k} inertia={km.inertia_:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(range(2, 16), inertia, marker='o', color='green', linewidth=2)
plt.title('Elbow Method')
plt.xlabel('k')
plt.ylabel('inertia')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# pca to visualize in 2d - doing this once and reusing
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# made so that there isnt repeats within the code of the same function 
def pca_plot(title, filename, labels=None, noise_mask=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    if noise_mask is not None:
        ax.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1], c='lightgrey', alpha=0.3, s=8)
        m = ~noise_mask
        ax.scatter(X_pca[m, 0], X_pca[m, 1],
                  c=[point_colors[i] for i in range(len(point_colors)) if m[i]], alpha=0.4, s=8)
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=point_colors, alpha=0.4, s=8)
    ax.legend(handles=patches, loc='upper right', fontsize=9, title='genre')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

k = 8

# kmeans
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_score = silhouette_score(X_scaled, kmeans_labels, sample_size=5000, random_state=42)
print("kmeans:", round(kmeans_score, 4))
pca_plot(f'K-Means (K={k})', 'pca_kmeans.png')

# kmeans++
kmeanspp = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
kmeanspp_labels = kmeanspp.fit_predict(X_scaled)
kmeanspp_score = silhouette_score(X_scaled, kmeanspp_labels, sample_size=5000, random_state=42)
print("kmeans++:", round(kmeanspp_score, 4))
pca_plot(f'K-Means++ (K={k})', 'pca_kmeanspp.png')

# kmedoids
kmedoids = KMedoids(n_clusters=k, random_state=42)
kmedoids_labels = kmedoids.fit_predict(X_scaled)
kmedoids_score = silhouette_score(X_scaled, kmedoids_labels, sample_size=5000, random_state=42)
print("kmedoids:", round(kmedoids_score, 4))
pca_plot(f'K-Medoids (K={k})', 'pca_kmedoids.png')

# dbscan - doesnt need k, groups by density
dbscan = DBSCAN(eps=0.8, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)
n_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"dbscan: {n_dbscan} clusters, {n_noise} noise points")
if n_dbscan > 1:
    m = dbscan_labels != -1
    dbscan_score = silhouette_score(X_scaled[m], dbscan_labels[m], sample_size=5000, random_state=42)
    print("dbscan score:", round(dbscan_score, 4))
else:
    dbscan_score = 0
noise_mask = np.array(dbscan_labels == -1)
pca_plot(f'DBSCAN ({n_noise} noise points in grey)', 'pca_dbscan.png', noise_mask=noise_mask)

# gmm 
gmm = GaussianMixture(n_components=k, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
gmm_score = silhouette_score(X_scaled, gmm_labels, sample_size=5000, random_state=42)
print("gmm:", round(gmm_score, 4))
pca_plot(f'GMM (K={k})', 'pca_gmm.png')

# hierarchical - using a sample because its slow on the full dataset
sidx = np.random.choice(len(X_scaled), 5000, replace=False)
X_s = X_scaled[sidx]
pca_s = X_pca[sidx]
g_s = [genres[i] for i in sidx]
c_s = [colors[g] for g in g_s]

hierarchical = AgglomerativeClustering(n_clusters=k)
h_labels = hierarchical.fit_predict(X_s)
hierarchical_score = silhouette_score(X_s, h_labels)
print("hierarchical:", round(hierarchical_score, 4))

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(pca_s[:, 0], pca_s[:, 1], c=c_s, alpha=0.5, s=8)
ax.legend(handles=patches, loc='upper right', fontsize=9, title='genre')
ax.set_title('Hierarchical Clustering (5k sample)')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('pca_hierarchical.png', dpi=150, bbox_inches='tight')
plt.show()

# all 6 tests in one figure
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()
all_labels = [kmeans_labels, kmeanspp_labels, kmedoids_labels, dbscan_labels, gmm_labels, None]
titles = [
    f'K-Means ({kmeans_score:.4f})',
    f'K-Means++ ({kmeanspp_score:.4f})',
    f'K-Medoids ({kmedoids_score:.4f})',
    f'DBSCAN ({n_dbscan} clusters, {n_noise} noise)',
    f'GMM ({gmm_score:.4f})',
    f'Hierarchical ({hierarchical_score:.4f}, 5k sample)'
]

for i in range(5):
    if 'DBSCAN' in titles[i]:
        noise = np.array(all_labels[i]) == -1
        axes[i].scatter(X_pca[noise, 0], X_pca[noise, 1], c='lightgrey', alpha=0.2, s=4)
        cm = ~noise
        axes[i].scatter(X_pca[cm, 0], X_pca[cm, 1],
                       c=[point_colors[j] for j in range(len(point_colors)) if cm[j]], alpha=0.4, s=4)
    else:
        axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=point_colors, alpha=0.4, s=4)
    axes[i].set_title(titles[i], fontsize=12, pad=8)
    axes[i].set_xlabel('PC1', fontsize=9)
    axes[i].set_ylabel('PC2', fontsize=9)
    axes[i].grid(alpha=0.2)
    axes[i].set_xlim(-11, 5)
    axes[i].set_ylim(-7, 5)

axes[5].scatter(pca_s[:, 0], pca_s[:, 1], c=c_s, alpha=0.5, s=4)
axes[5].set_title(titles[5], fontsize=12, pad=8)
axes[5].set_xlabel('PC1', fontsize=9)
axes[5].set_ylabel('PC2', fontsize=9)
axes[5].grid(alpha=0.2)
axes[5].set_xlim(-11, 5)
axes[5].set_ylim(-7, 5)

fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=11, title='genre', bbox_to_anchor=(0.5, -0.02))
plt.suptitle('All 6 Algorithms — Songs Colored by Genre', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('pca_all_algorithms.png', dpi=150, bbox_inches='tight')
plt.show()

# silhouette bar chart
algs = ['K-Means', 'K-Means++', 'K-Medoids', 'GMM', 'Hierarchical']
scores = [kmeans_score, kmeanspp_score, kmedoids_score, gmm_score, hierarchical_score]

plt.figure(figsize=(9, 5))
bars = plt.bar(algs, scores, color=['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c'])
plt.title('Silhouette Scores')
plt.ylabel('score')
plt.ylim(0, max(scores) + 0.05)
for bar, s in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{s:.4f}', ha='center', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('silhouette_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# heatmap of average features per cluster
df_results = X.copy()
df_results['cluster'] = kmeans_labels
df_results['genre'] = genres.values

cluster_avg = df_results.groupby('cluster')[features].mean().round(3)
plt.figure(figsize=(13, 6))
sns.heatmap(cluster_avg.T, annot=True, fmt='.2f', cmap='YlGn', linewidths=0.4)
plt.title('Avg Audio Features per Cluster (K-Means)')
plt.ylabel('feature')
plt.xlabel('cluster')
plt.tight_layout()
plt.savefig('cluster_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# top genres per cluster
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i in range(k):
    top = df_results[df_results['cluster'] == i]['genre'].value_counts().head(5)
    axes[i].bar(top.index, top.values, color='steelblue')
    axes[i].set_title(f'cluster {i}')
    axes[i].set_ylabel('count')
    axes[i].tick_params(axis='x', rotation=30)
plt.suptitle('Top Genres per Cluster', fontsize=14)
plt.tight_layout()
plt.savefig('genre_bar_chart.png', dpi=150, bbox_inches='tight')
plt.show()

print("done")
print(f"kmeans: {kmeans_score:.4f}, kmeans++: {kmeanspp_score:.4f}, kmedoids: {kmedoids_score:.4f}")
print(f"gmm: {gmm_score:.4f}, hierarchical: {hierarchical_score:.4f}")
print(f"dbscan: {n_dbscan} clusters, {n_noise} noise")
