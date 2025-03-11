import faiss
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Descargar stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("spanish"))

# 1️⃣ Cargar modelos y datos
CONFIG = {
    "faiss_index_path": "faiss_index.bin",
    "categories_path": "categories.pkl",
    "model_name": "distiluse-base-multilingual-cased-v2",
    "ef_search": 100,  # Precisión en búsqueda
    "top_k": 5  # Número de resultados a devolver
}

print("⏳ Cargando modelo y datos...")
index = faiss.read_index(CONFIG["faiss_index_path"])
categories = pickle.load(open(CONFIG["categories_path"], "rb"))
model = SentenceTransformer(CONFIG["model_name"])
print("✅ Modelos y datos cargados.")

# Validar que FAISS y categories.pkl tengan el mismo tamaño
print(f"FAISS contiene: {index.ntotal} embeddings")
print(f"Categorías almacenadas: {len(categories)}")
if index.ntotal != len(categories):
    print("⚠️ Advertencia: El número de embeddings en FAISS y las categorías no coinciden. Esto puede causar errores en las búsquedas.")

# Función para limpiar el texto antes de la búsqueda
def clean_text(text):
    """Normaliza el texto eliminando caracteres especiales, convirtiendo a minúsculas y eliminando espacios extra."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9áéíóúüñ ]", "", text)  # Eliminar caracteres especiales
    text = re.sub(r"\s+", " ", text).strip()  # Eliminar espacios extra
    text = " ".join([word for word in text.split() if word not in stop_words])  # Eliminar stopwords
    return text

# 2️⃣ Configurar API
app = FastAPI()

class QueryModel(BaseModel):
    text: str

@app.post("/search")
def search(query: QueryModel):
    """Busca los productos más similares y devuelve las categorías y subcategorías con su porcentaje de certeza."""
    cleaned_text = clean_text(query.text)  # Aplicar limpieza al texto de búsqueda
    query_vector = model.encode([cleaned_text], convert_to_numpy=True)
    
    distances, indices = index.search(query_vector, CONFIG["top_k"])
    print("🔍 Resultados FAISS:", indices, distances)  # Validar salida de FAISS
    
    results = []
    
    if len(indices) > 0:
        for j, i in enumerate(indices[0]):
            if 0 <= i < len(categories):  # Validar que el índice está dentro del rango
                confidence = round((1 / (1 + distances[0][j])) * 100, 2)  # Convertir distancia en % de certeza
                print(f"✔️ Índice válido: {i}, Certeza: {confidence}%")
                results.append({
                    "category": categories[i][0],
                    "subcategory": categories[i][1],
                    "confidence": confidence
                })
            else:
                print(f"❌ Índice fuera de rango: {i}")
    else:
        print("⚠️ No se encontraron resultados en FAISS.")
    
    return {"query": query.text, "results": results}

# 3️⃣ Ejecutar API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=CONFIG.get("api_port", 8000))