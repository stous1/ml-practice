import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# 1. Generar un dataset sintético
np.random.seed(42)
n_samples = 1000

# Generar variables
age = np.random.randint(18, 70, size=n_samples)                   # Edad del usuario
gender = np.random.choice([0, 1], size=n_samples)                   # 0: Femenino, 1: Masculino
time_sent = np.random.randint(0, 24, size=n_samples)                # Hora de envío (0 a 23)
day_of_week = np.random.randint(0, 7, size=n_samples)               # Día de la semana (0=Lunes, 6=Domingo)
subject_length = np.random.randint(5, 50, size=n_samples)           # Longitud del asunto (en número de palabras)

# Variable indicadora: si el email se envió en fin de semana (sábado o domingo)
weekend = (day_of_week >= 5).astype(int)

# Crear un DataFrame con las variables
df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'time_sent': time_sent,
    'subject_length': subject_length,
    'weekend': weekend
})

# 2. Definir la probabilidad "real" de apertura usando una función logística
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Definir una combinación lineal con pesos arbitrarios:
# - Un asunto más largo (subject_length) puede aumentar la probabilidad (peso 0.05)
# - Una mayor edad puede disminuir la probabilidad (peso -0.02)
# - El género puede tener un pequeño efecto (peso 0.1)
# - La hora de envío puede influir (por ejemplo, emails enviados en horas "laborables" pueden tener mayor probabilidad; usamos 0.05)
# - Enviar en fin de semana puede aumentar la probabilidad (peso 0.3)
linear_combination = (
    0.05 * df['subject_length'] -
    0.02 * df['age'] +
    0.1  * df['gender'] +
    0.05 * df['time_sent'] +
    0.3  * df['weekend']
)

# Calcular la probabilidad de apertura con la función sigmoide
prob_open = sigmoid(linear_combination)

# Generar la variable objetivo "opened" (1: abrió el email, 0: no abrió)
opened = np.random.binomial(1, prob_open)
df['opened'] = opened

# Visualizar las primeras filas del dataset
print("Dataset Sintético:")
print(df.head())

# 3. Preparar los datos para el modelo
features = ['age', 'gender', 'time_sent', 'subject_length', 'weekend']
X = df[features]
y = df['opened']

# Dividir en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Entrenar el modelo de regresión logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Calcular las probabilidades de apertura en el conjunto de prueba
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluar el modelo usando ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("\nROC AUC del modelo:", roc_auc)

# Mostrar algunas predicciones de probabilidad
predicciones = X_test.copy()
predicciones['prob_apertura'] = y_pred_prob
print("\nPredicciones (primeros 5 ejemplos):")
print(predicciones.head())
