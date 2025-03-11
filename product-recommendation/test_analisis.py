import pandas as pd
from sklearn.metrics import confusion_matrix

# Cargar el archivo CSV
df = pd.read_csv('test_results.csv')

# Mostrar las primeras filas del DataFrame
print("Primeras filas del DataFrame:")
print(df.head())

# Mostrar información general del DataFrame
print("\nInformación general del DataFrame:")
print(df.info())

# Descripción estadística de las columnas numéricas
print("\nDescripción estadística de las columnas numéricas:")
print(df.describe())

# Verificar si hay valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Mostrar la distribución de las columnas categóricas
print("\nDistribución de las columnas categóricas:")
for column in df.select_dtypes(include=['object']).columns:
    print(f"\n{column}:\n{df[column].value_counts()}")
    
    # Suponiendo que las columnas 'actual' y 'predicted' contienen las categorías reales y predichas
    actual = df['true_category']
    predicted = df['predicted_category']

    # Calcular la matriz de confusión
    cm = confusion_matrix(actual, predicted)

    # Mostrar la matriz de confusión
    print("\nMatriz de confusión:")
    print(cm)