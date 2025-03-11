import requests
import pandas as pd

# Configuración de la API
API_URL = "http://localhost:8000/search"
EXCEL_PATH = "products.xlsx"

# Cargar datos del Excel
df = pd.read_excel(EXCEL_PATH, usecols=["prod_name", "subcategory", "tags"]).sample(n=100, random_state=42)  # Seleccionar 1000 valores aleatorios

# Función para probar la API con cada producto
def test_api():
    results = []
    
    for _, row in df.iterrows():
        product_name = row["prod_name"]
        true_category = row["subcategory"]
        true_subcategory = row["tags"]
        
        # Enviar consulta a la API
        response = requests.post(API_URL, json={"text": product_name + " - " + true_category + " - " + true_subcategory})
        if response.status_code == 200:
            data = response.json()
            predicted_results = data.get("results", [])
            
            if predicted_results:
                best_match = predicted_results[0]  # Tomar la mejor coincidencia
                predicted_category = best_match["category"]
                predicted_subcategory = best_match["subcategory"]
                confidence = best_match["confidence"]
                
                # Evaluar si la predicción fue correcta
                is_correct = (predicted_category == true_category and predicted_subcategory == true_subcategory)
                
                results.append({
                    "prod_name": product_name,
                    "true_category": true_category,
                    "true_subcategory": true_subcategory,
                    "predicted_category": predicted_category,
                    "predicted_subcategory": predicted_subcategory,
                    "confidence": confidence,
                    "correct": is_correct
                })
            else:
                results.append({
                    "prod_name": product_name,
                    "true_category": true_category,
                    "true_subcategory": true_subcategory,
                    "predicted_category": "N/A",
                    "predicted_subcategory": "N/A",
                    "confidence": 0,
                    "correct": False
                })
        else:
            print(f"⚠️ Error en la API para '{product_name}': {response.status_code}")
    
    # Convertir resultados a DataFrame y mostrar estadísticas
    results_df = pd.DataFrame(results)
    accuracy = results_df["correct"].mean() * 100
    print(f"🔍 Precisión del modelo: {accuracy:.2f}%")
    print(results_df.head(10))  # Mostrar primeras filas
    
    # Guardar resultados en CSV
    results_df.to_csv("test_results.csv", index=False)
    print("✅ Resultados guardados en 'test_results.csv'")

if __name__ == "__main__":
    test_api()
