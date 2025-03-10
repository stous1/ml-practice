import mlflow
import mlflow.h2o
import h2o
from h2o.automl import H2OAutoML
from sklearn.datasets import load_iris
import pandas as pd

# 1. Inicializa H2O
h2o.init()

# 2. Configura MLflow para usar un experimento llamado "AutoML_Basico_Todos"
mlflow.set_experiment("AutoML_Basico_Todos")

# 3. Carga el dataset Iris y conviértelo a un H2OFrame
iris = load_iris(as_frame=True)
df = iris.frame
hf = h2o.H2OFrame(df)
hf["target"] = hf["target"].asfactor()  # Para clasificación

features = iris.feature_names
target = "target"

# 4. Ejecuta H2O AutoML (limitamos a 3 modelos o 30 segundos)
aml = H2OAutoML(max_models=10, max_runtime_secs=600, seed=1)
aml.train(x=features, y=target, training_frame=hf)

# 5. Obtén la leaderboard en forma de DataFrame de Pandas
lb = aml.leaderboard.as_data_frame()
print("Leaderboard:")
print(lb)

# 6. Itera sobre cada modelo en la leaderboard y regístralo en MLflow
for model_id in lb["model_id"]:
    h2o_model = h2o.get_model(model_id)
    
    # Obtener el rendimiento del modelo
    perf = h2o_model.model_performance()

    # Para clasificación multiclase (como Iris), calcular precisión como 1 - mean_per_class_error
    accuracy = 1 - perf.mean_per_class_error()
    
    with mlflow.start_run(run_name=f"Model_{model_id}") as run:
        mlflow.log_param("model_id", model_id)
        if accuracy is not None:
            mlflow.log_metric("accuracy", accuracy)
        
        # Guarda el modelo en MLflow
        mlflow.h2o.log_model(h2o_model, f"modelo_{model_id}")
        print(f"Modelo {model_id} guardado en MLflow en run_id: {run.info.run_id}")
