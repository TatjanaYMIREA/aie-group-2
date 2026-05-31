import joblib
import json
from pathlib import Path

_model = None
_features = None

def load_model(model_path: str = "./artifacts/happiness_model.pkl"):
    global _model, _features
    _model = joblib.load(model_path)
    with open("./artifacts/features.json", 'r') as f:
        _features = json.load(f)
    print(f"Модель загружена из {model_path}")

def predict(features_dict: dict) -> float:
    global _model, _features
    if _model is None:
        raise RuntimeError("Модель не загружена. Вызовите load_model()")
    
    # Преобразуем ключи (Log_GDP_per_capita -> Log GDP per capita)
    model_features = {}
    for key, value in features_dict.items():
        model_key = key.replace('_', ' ')
        model_features[model_key] = value
    
    feature_values = [model_features[f] for f in _features]
    return float(_model.predict([feature_values])[0])
