import logging
import time
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from .schemas import HappinessInput, HappinessOutput
from .predict import load_model, predict

# ============================================
# НАСТРОЙКА ЛОГИРОВАНИЯ (консоль + файл)
# ============================================

# Создаём папку для логов, если её нет
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Имя файла лога с текущей датой
log_filename = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")

# Настройка логгера
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Формат логов
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Хэндлер для консоли
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Хэндлер для файла
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Добавляем оба хэндлера
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Отключаем propagate, чтобы логи не дублировались
logger.propagate = False

# ============================================
# FASTAPI ПРИЛОЖЕНИЕ
# ============================================

app = FastAPI(title="Happiness Prediction API", version="1.0.0")

# Middleware для логирования всех запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - Время: {process_time:.3f}s")
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Загружаем модель при старте
load_model()
logger.info("Сервис запущен")

@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "ok", "model_loaded": True}

@app.post(
    "/predict",
    response_model=HappinessOutput,
    responses={
        200: {"description": "Успешное предсказание", "model": HappinessOutput},
        422: {"description": "Ошибка валидации входных данных"},
        500: {"description": "Внутренняя ошибка сервера"}
    }
)
async def predict_happiness(data: HappinessInput):
    try:
        input_dict = data.model_dump()
        logger.info(f"Получен запрос на предсказание: {input_dict}")
        
        result = predict(input_dict)
        logger.info(f"Предсказание: {result}")
        
        return HappinessOutput(life_ladder=round(result, 3))
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
