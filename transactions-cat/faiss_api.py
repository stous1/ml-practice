from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Inicializar FastAPI
app = FastAPI()

# Modelo de embeddings
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

# Ruta del archivo de índice FAISS y categorías
faiss_index_path = os.path.join("faiss_index.bin")
categories_path = os.path.join("categories.txt")

# Verificar si el archivo de índice existe y cargarlo
if os.path.exists(faiss_index_path):
    print("Cargando índice FAISS desde archivo...")
    index = faiss.read_index(faiss_index_path)
else:
    raise FileNotFoundError(f"El archivo de índice FAISS no se encontró en {faiss_index_path}")

# Cargar las categorías desde el archivo
if os.path.exists(categories_path):
    with open(categories_path, "r") as f:
        category_names = [line.strip() for line in f.readlines()]
else:
    raise FileNotFoundError(f"El archivo de categorías no se encontró en {categories_path}")

# Modelo Pydantic para recibir input
class TransactionInput(BaseModel):
    description: str

@app.post("/categorize")
def categorize_transaction(transaction: TransactionInput):
    # Generar embedding para la descripción ingresada
    query_embedding = np.array([model.encode(transaction.description)], dtype="float32")

    # Buscar la categoría más cercana en FAISS
    index.hnsw.efSearch = 100
    _, indices = index.search(query_embedding, 1)
    best_category = category_names[indices[0][0]]

    return {"category": best_category}