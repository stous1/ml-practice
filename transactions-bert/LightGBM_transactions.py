import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Cargar dataset
file_path = "C:/Users/sergi/projects/ml-practice/transactions-bert/dataset_transacciones_colombia.csv"
df = pd.read_csv(file_path)

# Tokenizar descripciones
df["tokens"] = df["Descripcion"].apply(lambda x: word_tokenize(str(x).lower()))

# Entrenar Word2Vec
w2v_model = Word2Vec(sentences=df["tokens"], vector_size=100, window=5, min_count=2, workers=4)

def text_to_vec(text):
    vectors = [w2v_model.wv[word] for word in text if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

# Convertir texto a vectores
df["vector"] = df["tokens"].apply(text_to_vec)
X = np.vstack(df["vector"].values)

# Mapear categorías a índices
categorias = df["Categoria"].unique()
categoria_to_idx = {cat: idx for idx, cat in enumerate(categorias)}
df["Categoria_ID"] = df["Categoria"].map(categoria_to_idx)
y = df["Categoria_ID"].values

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Configuración de LightGBM con hiperparámetros optimizados
params = {
    "objective": "multiclass",
    "num_class": len(categorias),
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42
}

# Entrenamiento con validación cruzada
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for train_idx, val_idx in kf.split(X_train, y_train):
    train_data = lgb.Dataset(X_train[train_idx], label=y_train[train_idx])
    val_data = lgb.Dataset(X_train[val_idx], label=y_train[val_idx], reference=train_data)
    model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[val_data], early_stopping_rounds=50, verbose_eval=100)
    y_pred = model.predict(X_train[val_idx])
    y_pred_labels = [x.argmax() for x in y_pred]
    accuracy = accuracy_score(y_train[val_idx], y_pred_labels)
    accuracies.append(accuracy)
    print(f"Fold Accuracy: {accuracy:.4f}")

print(f"Mean CV Accuracy: {np.mean(accuracies):.4f}")

# Evaluación final en test set
y_pred_test = model.predict(X_test)
y_pred_test_labels = [x.argmax() for x in y_pred_test]
print("Test Accuracy:", accuracy_score(y_test, y_pred_test_labels))

# Guardar el modelo
model.save_model("lightgbm_transactions_model.txt")
