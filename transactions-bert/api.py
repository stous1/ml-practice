from fastapi import FastAPI
import lightgbm as lgb
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")

# Inicializar la API
app = FastAPI()

# Cargar el modelo entrenado
model = lgb.Booster(model_file="lightgbm_transactions_model.txt")

# Cargar el modelo de Word2Vec
w2v_model = Word2Vec.load("word2vec_model.bin")  # Asegúrate de haber guardado este modelo antes

# Mapeo de categorías
categorias = ["Alimentación", "Transporte", "Salud", "Hogar", "Ocio"]  # Modifica según tu dataset
idx_to_categoria = {i: cat for i, cat in enumerate(categorias)}

def text_to_vec(text):
    tokens = word_tokenize(text.lower())
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

# Endpoint para predecir la categoría de una transacción
@app.post("/predict/")
def predict_transaction(description: str):
    vector = text_to_vec(description).reshape(1, -1)
    prediction = model.predict(vector)
    category_idx = np.argmax(prediction)
    category = idx_to_categoria.get(category_idx, "Desconocido")

    return {"descripcion": description, "categoria_predicha": category}

# Ejecutar con: uvicorn nombre_archivo:app --reload
