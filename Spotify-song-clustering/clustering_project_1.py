# Step-1 Data pre-processing.

import pandas as pd
df = pd.read_csv("c:/Users/Vishnu/Downloads/Compressed/rolling_stones_spotify.csv")

# check for duplicates
df.duplicated().sum()          
# drop duplicate rows
df = df.drop_duplicates()  

# count missing values in each column    
df.isnull().sum()   
# replacing missing values with column mean
df = df.fillna(df.mean(numeric_only=True))

# Drop irrelevant columns
df = df.drop(columns=["id", "uri", "name", "album", "release_date", "track_number", df.columns[0]])

# Applying Z score to remove outliers
z = (df - df.mean()) / df.std()

# removing outliers
df = df[(abs(z) < 3).all(axis=1)]


# normalizing 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# step2 elbow plot
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    inertia.append(kmeans.inertia_)

plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# 5 is where approximately elbow plot flattened. so k=5
# step 3 clustering

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(scaled_df)

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


# Step 4 Run t-SNE for visualization
# I tried PCA but TSNE gave better plot of clusters
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)

tsne_result = tsne.fit_transform(scaled_df)

# Create DataFrame for plotting
tsne_df = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])
tsne_df['cluster'] = kmeans.labels_

# Plot
sns.scatterplot(x='TSNE1', y='TSNE2', hue='cluster', data=tsne_df, palette='Set2')
plt.title("t-SNE Clustering Visualization")
plt.show()

df['cluster'] = kmeans.labels_
cluster_summary = df.groupby('cluster').mean(numeric_only=True)
print(cluster_summary)

