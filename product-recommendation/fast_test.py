import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 1️⃣ Cargar el dataset
file_path = "products.xlsx"
df = pd.read_excel(file_path)  # Archivo separado por tabulaciones

# 2️⃣ Procesar los datos y seleccionar campos clave
df["text_representation"] = df["prod_name"] + " - " + df["category"] + " - " + df["subcategory"]
products = df["text_representation"].tolist()

# 3️⃣ Generar embeddings
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
embeddings = np.array([model.encode(p) for p in products], dtype=np.float32)

# 4️⃣ Reducir dimensionalidad con PCA para mejorar eficiencia
d = embeddings.shape[1]  # Dimensión original
pca_dim = 128  # Reducir a 128 dimensiones
pca = faiss.PCAMatrix(d, pca_dim)
pca.train(embeddings)
reduced_embeddings = pca.apply(embeddings)

# 5️⃣ Configurar FAISS con HNSW
M = 32  # Número de conexiones por nodo
ef_construction = 100  # Precisión en construcción
ef_search = 50  # Precisión en búsqueda
index = faiss.IndexHNSWFlat(pca_dim, M, faiss.METRIC_L2)
index.hnsw.efConstruction = ef_construction
index.hnsw.efSearch = ef_search
index.add(reduced_embeddings)  # Agregar datos al índice

# 6️⃣ Crear API con FastAPI
app = FastAPI()

class QueryModel(BaseModel):
    text: str
    top_k: int = 5

@app.post("/search")
def search(query: QueryModel):
    query_vector = model.encode([query.text])
    query_vector = pca.apply(np.array(query_vector, dtype=np.float32))
    distances, indices = index.search(query_vector, query.top_k)
    results = [{"product": products[i], "distance": float(distances[0][j])} for j, i in enumerate(indices[0])]
    return {"query": query.text, "results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
