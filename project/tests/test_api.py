import pytest
from fastapi.testclient import TestClient
import sys
import os

# Добавляем путь к папке src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Правильный импорт — из happiness.api
from src.happiness.api import app

client = TestClient(app)


def test_health_check():
    """Тест health-check эндпоинта"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid():
    """Тест предсказания с корректными данными"""
    test_data = {
        "Log_GDP_per_capita": 9.5,
        "Social_support": 0.8,
        "Healthy_life_expectancy_at_birth": 65.0,
        "Freedom_to_make_life_choices": 0.7,
        "Generosity": 0.1,
        "Perceptions_of_corruption": 0.7,
        "Positive_affect": 0.6,
        "Negative_affect": 0.3
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "life_ladder" in response.json()
    assert isinstance(response.json()["life_ladder"], float)


def test_predict_invalid():
    """Тест предсказания с некорректными данными"""
    test_data = {
        "Log_GDP_per_capita": -1.0,
        "Social_support": 0.8,
        "Healthy_life_expectancy_at_birth": 65.0,
        "Freedom_to_make_life_choices": 0.7,
        "Generosity": 0.1,
        "Perceptions_of_corruption": 0.7,
        "Positive_affect": 0.6,
        "Negative_affect": 0.3
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422


def test_predict_missing_field():
    """Тест предсказания с отсутствующим полем"""
    test_data = {
        "Log_GDP_per_capita": 9.5,
        "Social_support": 0.8
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422