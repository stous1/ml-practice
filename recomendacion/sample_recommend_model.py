import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Cargar el dataset MovieLens 100k
# Asegúrate de tener el archivo 'u.data' descargado desde:
# https://files.grouplens.org/datasets/movielens/ml-100k/u.data
# y colócalo en una carpeta llamada 'ml-100k' en tu directorio de trabajo.

# El dataset tiene el formato: user_id, item_id, rating, timestamp (separado por tabulaciones)
df = pd.read_csv('recomendacion\data.csv', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
print("Dataset MovieLens 100k:")
print(df.head())

# 2. Construir la matriz usuario-ítem
user_item = df.pivot_table(index='user_id', columns='item_id', values='rating')
print("\nMatriz Usuario-Ítem (con NaN para valores no evaluados):")
print(user_item.head())

# 3. Calcular la media de ratings por usuario
user_means = user_item.mean(axis=1)
print("\nMedia de ratings por usuario:")
print(user_means.head())

# 4. Normalizar la matriz restando la media de cada usuario
norm_matrix = user_item.sub(user_means, axis=0)
print("\nMatriz de ratings normalizada (centrada):")
print(norm_matrix.head())

# 5. Rellenar los NaN con 0 para calcular la similitud
norm_matrix_filled = norm_matrix.fillna(0)

# 6. Calcular la similitud coseno entre usuarios
similarity = cosine_similarity(norm_matrix_filled)
similarity_df = pd.DataFrame(similarity, index=norm_matrix.index, columns=norm_matrix.index)
print("\nMatriz de similitud entre usuarios:")
print(similarity_df.head())

# 7. Función para predecir el rating de un usuario para un ítem (película) que no ha evaluado
def predict_rating(user, item, user_item, user_means, similarity_df):
    if item not in user_item.columns:
        # Si el ítem no existe en la matriz, retornamos la media global o del usuario
        return user_means[user]
    
    # Usuarios que han evaluado el ítem
    rated_users = user_item[item].dropna().index
    if len(rated_users) == 0:
        return user_means[user]
    
    numer = 0.0
    denom = 0.0
    for other in rated_users:
        if other == user:
            continue
        sim = similarity_df.loc[user, other]
        deviation = user_item.loc[other, item] - user_means[other]
        numer += sim * deviation
        denom += abs(sim)
    if denom == 0:
        return user_means[user]
    prediction = user_means[user] + numer / denom
    return prediction

# Ejemplo: Predecir el rating que el usuario 10 daría a la película 50 (si no la evaluó)
pred = predict_rating(10, 50, user_item, user_means, similarity_df)
print(f"\nPredicción para el usuario 10 en el ítem (película) 50: {pred:.2f}")

# 8. Función para recomendar ítems para un usuario dado
def recommend_items(user, user_item, user_means, similarity_df, top_n=5):
    # Ítems que el usuario no ha evaluado
    unrated = user_item.columns[user_item.loc[user].isna()]
    predictions = {}
    for item in unrated:
        predictions[item] = predict_rating(user, item, user_item, user_means, similarity_df)
    # Ordenamos los ítems por rating predicho (de mayor a menor)
    recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

# Ejemplo: Recomendar 5 películas para el usuario 10
recs = recommend_items(10, user_item, user_means, similarity_df, top_n=5)
print("\nRecomendaciones para el usuario 10:")
for item, rating in recs:
    print(f"Ítem {item} con rating predicho: {rating:.2f}")
