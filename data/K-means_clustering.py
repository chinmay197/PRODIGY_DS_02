# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(r"C:\Users\Chinmay\Desktop\PRODIGY_DS_02\data\Mall_Customers.csv")

# Display first few rows
print(df.head())

# Drop non-numeric or irrelevant columns (CustomerID, Gender for simplicity)
data = df.drop(['CustomerID', 'Gender'], axis=1)

# Optional: Standardize data for better clustering performance
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Use Elbow Method to find optimal number of clusters (k)
wcss = []  # Within-Cluster Sum of Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# From the elbow graph, choose k=5 (commonly optimal for this dataset)
k = 5
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# Add cluster labels to original dataframe
df['Cluster'] = cluster_labels

# Visualize clusters using 2 key features: Annual Income vs Spending Score
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Annual Income (k$)', y='Spending Score (1-100)',
    hue='Cluster', palette='Set2', data=df, s=100
)
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

