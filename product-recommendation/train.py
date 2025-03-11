import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import time

# Definir ruta del archivo excel y del índice FAISS
excel_path = "products.xlsx"
faiss_index_path = os.path.join("faiss_index.bin")
categories_path = os.path.join("categories.txt")

try:
    start_time = time.time()

    # Cargar el dataset
    print("Cargando dataset...")
    df = pd.read_excel(excel_path)
    print(f"Dataset cargado en {time.time() - start_time:.2f} segundos")

    # Eliminar columnas 'Fecha' y 'Codigo_Transaccion'
    df = df.drop(columns=["Fecha", "Codigo_Transaccion", "Monto_COP"])
    
    # Concatenar columnas 'Categoria' y 'Subcategoria'
    # df["Categoria"] = df["Categoria"] + " - " + df["Subcategoria"]
    # df = df.drop(columns=["Subcategoria"])
    
    # Mantener solo registros únicos
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Contar el número de categorías únicas
    num_categories = df["Categoria"].nunique()
    print(f"Número de categorías únicas: {num_categories}")
    
    # Inicializar modelo de embeddings
    print("Generando embeddings...")
    model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    embeddings_start_time = time.time()
    embeddings = model.encode(df["Descripcion"].tolist(), convert_to_numpy=True)
    print(f"Embeddings generados en {time.time() - embeddings_start_time:.2f} segundos")
    
    print("Creando nuevo índice FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32)  # 32 es el número de vecinos en el grafo
    
    # Añadir embeddings al índice
    add_start_time = time.time()
    index.add(embeddings)
    print(f"Embeddings añadidos al índice en {time.time() - add_start_time:.2f} segundos")
    
    # Guardar el índice FAISS en un archivo
    save_start_time = time.time()
    print("Guardando índice FAISS...")
    
    if os.path.exists(faiss_index_path):
        os.remove(faiss_index_path)
        
    faiss.write_index(index, faiss_index_path)
    print(f"Índice FAISS guardado en {time.time() - save_start_time:.2f} segundos")

    # Guardar las categorías en un archivo
    print("Guardando categorías en archivo...")
    with open(categories_path, "w") as f:
        for desc in df["Categoria"].tolist():
            f.write(desc + "\n")
    print(f"Categorías guardadas en {categories_path}")

    total_time = time.time() - start_time
    print(f"Índice FAISS listo y guardado en {faiss_index_path} en {total_time:.2f} segundos")

except Exception as e:
    print(f"Error: {e}")