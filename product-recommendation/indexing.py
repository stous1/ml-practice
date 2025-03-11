import faiss
import pandas as pd
import numpy as np
import os
import time
import pickle
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords

# Descargar stopwords si es necesario
nltk.download("stopwords")
stop_words = set(stopwords.words("spanish"))

# 1️⃣ Configuración de parámetros
CONFIG = {
    "excel_path": "products.xlsx",
    "faiss_index_path": "faiss_index.bin",
    "categories_path": "categories.pkl",
    "model_name": "distiluse-base-multilingual-cased-v2",
    "embedding_dim": 512,  # Dimensión después de PCA (si se usa)
    "hnsw_M": 64,  # Número de conexiones por nodo en HNSW
    "ef_construction": 200,  # Precisión en construcción
    "ef_search": 200,  # Precisión en búsqueda
    "columns_to_keep": ["prod_name", "subcategory", "tags"],  # Columnas a conservar
    "text_representation_format": "{prod_name} - {subcategory} - {tags}"  # Formato de texto para embeddings
}

# Inicializar modelo de embeddings
model = SentenceTransformer(CONFIG["model_name"])


def clean_text(text):
    """Normaliza el texto eliminando caracteres especiales, stopwords y espacios extra."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9áéíóúüñ ]", "", text)  # Eliminar caracteres especiales
    text = re.sub(r"\s+", " ", text).strip()  # Eliminar espacios extra
    text = " ".join([word for word in text.split() if word not in stop_words])  # Eliminar stopwords
    return text


def load_dataset():
    """Carga el dataset desde un archivo Excel y realiza preprocesamiento."""
    print("⏳ Cargando dataset...")
    df = pd.read_excel(CONFIG["excel_path"], usecols=CONFIG["columns_to_keep"])
    df = df.drop_duplicates().reset_index(drop=True)
    df["text_representation"] = df.apply(lambda row: clean_text(CONFIG["text_representation_format"].format(**row)), axis=1)
    print(f"✅ Dataset cargado con {len(df)} registros únicos.")
    return df


def generate_embeddings(texts):
    """Genera embeddings para una lista de textos usando SentenceTransformer con batch processing."""
    print("⏳ Generando embeddings...")
    embeddings = model.encode(list(tqdm(texts)), convert_to_numpy=True)
    print("✅ Embeddings generados con dimensión:", embeddings.shape[1])
    return embeddings


def create_faiss_index(embeddings, dimension):
    """Crea un índice FAISS con HNSW y añade los embeddings."""
    print("⏳ Creando índice FAISS con HNSW...")
    if dimension != CONFIG["embedding_dim"]:
        print(f"⚠️ Advertencia: La dimensión de los embeddings ({dimension}) no coincide con la configuración ({CONFIG['embedding_dim']}).")
    index = faiss.IndexHNSWFlat(dimension, CONFIG["hnsw_M"], faiss.METRIC_L2)
    index.hnsw.efConstruction = CONFIG["ef_construction"]
    index.hnsw.efSearch = CONFIG["ef_search"]
    index.add(embeddings)
    print("✅ Índice FAISS construido.")
    return index


def save_faiss_index(index):
    """Guarda el índice FAISS en un archivo binario."""
    print("⏳ Guardando índice FAISS...")
    faiss.write_index(index, CONFIG["faiss_index_path"])
    print(f"✅ Índice FAISS guardado en {CONFIG['faiss_index_path']}.")


def save_categories(categories):
    """Guarda las categorías en un archivo pickle."""
    print("⏳ Guardando categorías...")
    with open(CONFIG["categories_path"], "wb") as f:
        pickle.dump(categories, f)
    print(f"✅ Categorías guardadas en {CONFIG['categories_path']}.")


if __name__ == "__main__":
    start_time = time.time()

    # Cargar y procesar dataset
    df = load_dataset()
    categories = df[["subcategory", "tags"]].values.tolist()  # Mantener alineación con embeddings

    # Generar embeddings
    embeddings = generate_embeddings(df["text_representation"].tolist())
    
    # Validación de coherencia entre embeddings y categorías
    if len(categories) != len(embeddings):
        print(f"⚠️ Advertencia: Número de embeddings ({len(embeddings)}) y categorías ({len(categories)}) no coincide. Esto puede causar errores en la API.")
    
    # Crear índice FAISS
    dimension = embeddings.shape[1]
    index = create_faiss_index(embeddings, dimension)
    
    # Guardar índice FAISS y categorías
    save_faiss_index(index)
    save_categories(categories)

    print(f"🎉 Proceso completado en {time.time() - start_time:.2f} segundos.")
