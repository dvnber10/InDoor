from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import math

app = FastAPI()

# === MODELO Y COMPONENTES PCA ===

# Cargar modelo Random Forest
with open("modelo_entrenado_rf.pkl", "rb") as f:
    forest_trees = pickle.load(f)

# Define estructura de RandomForest igual al entrenamiento
class RandomForest:
    def __init__(self, trees):
        self.trees = trees

    def predict_tree(self, node, row):
        if float(row[node['index']]) < node['value']:
            if isinstance(node['left'], dict):
                return self.predict_tree(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_tree(node['right'], row)
            else:
                return node['right']

    def predict(self, row):
        predictions = [self.predict_tree(tree, row) for tree in self.trees]
        return max(set(predictions), key=predictions.count)

model = RandomForest(forest_trees)

# PCA COMPONENTES (debes pegarlos aquí si los tienes como listas)
pca_components = [
    
    [-0.6166099557461893, 0.567591736464548, 1.0421217153826856e-17],  
    [-0.6166099557461893, 0.567591736464548, 1.0421217153826856e-17],  
    [-0.6166099557461893, 0.567591736464548, 1.0421217153826856e-17]   
]

# Normalización y centrado (valores obtenidos antes)
min_vals = [-0.6166099557461893] * 3
max_vals = [0.567591736464548] * 3
means = [1.0421217153826856e-17] * 3

# === INPUT FORMAT ===
class SensorInput(BaseModel):
    sensor1: list[float]  # 3 valores
    sensor2: list[float]  # 3 valores
    sensor3: list[float]  # 3 valores

# === UTILS ===
def flatten_sensors(data: SensorInput):
    return data.sensor1 + data.sensor2 + data.sensor3  # 9 valores

def project_pca(x, components):
    return [sum(x[i] * v[i] for i in range(len(x))) for v in components]

def normalize(x):
    return [(x[i] - min_vals[i]) / (max_vals[i] - min_vals[i]) if max_vals[i] != min_vals[i] else 0 for i in range(len(x))]

def center(x):
    return [x[i] - means[i] for i in range(len(x))]

# === ENDPOINT ===
@app.post("/predecir")
def predecir(data: SensorInput):
    original = flatten_sensors(data)  # 9D
    x_pca = project_pca(original, pca_components)  # 3D
    x_norm = normalize(x_pca)
    x_centered = center(x_norm)
    pred = model.predict(x_centered)
    return {"prediccion": pred}
