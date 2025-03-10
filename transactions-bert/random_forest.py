from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Cargar dataset
df = pd.read_csv("C:/Users/sergi/projects/ml-practice/transactions-bert/dataset_transacciones_colombia.csv")

# Dividir datos
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Descripcion"], df["Categoria"], test_size=0.2, random_state=42
)

# Crear modelo con TF-IDF + Random Forest
pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100))
pipeline.fit(train_texts, train_labels)

# Evaluar modelo
preds = pipeline.predict(val_texts)
accuracy = accuracy_score(val_labels, preds)
print(f"Accuracy: {accuracy:.4f}")
