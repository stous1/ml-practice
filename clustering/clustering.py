import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# 1. Cargar el dataset Iris
iris = load_iris()
X = iris.data  # Características
feature_names = iris.feature_names

# Convertir a DataFrame para mayor claridad
df = pd.DataFrame(X, columns=feature_names)

# 2. Aplicar K-Means para agrupar los datos en 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
df['cluster'] = clusters

# 3. Mostrar los centros de cada cluster
print("Centros de los clusters:")
print(kmeans.cluster_centers_)

# 4. Mostrar las primeras filas con la asignación de clusters
print("\nPrimeras filas con clusters asignados:")
print(df.head())

# 5. Visualizar los clusters en función de dos características (por ejemplo, petal length y petal width)
plt.figure(figsize=(8, 6))
plt.scatter(df[feature_names[2]], df[feature_names[3]], c=df['cluster'], cmap='viridis', edgecolor='k', s=50)
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[3])
plt.title("Clusters en el dataset Iris (K-Means)")
plt.show()
