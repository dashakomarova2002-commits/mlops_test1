# mlops_test1
# MLOps Project: Titanic Survival Prediction

Проект демонстрирует полный цикл MLOps: от обучения модели на данных Kaggle до упаковки в Docker-контейнер и развертывания REST API.

## Технологии
* **Python 3.9+**, Scikit-learn, Pandas
* **MLflow** (трекинг экспериментов)
* **FastAPI**, Uvicorn (веб-сервер)
* **Docker** (контейнеризация)

## 📂 Структура проекта
* `train.py` — скрипт обучения модели (RandomForest) и логирования в MLflow.
* `app.py` — FastAPI приложение для инференса модели.
* `requirements.txt` — зависимости Python.
* `Dockerfile` — инструкция для сборки Docker-образа.
* `test_request.py` — скрипт для тестирования API.

## Как запустить

### 1. Обучение модели и MLflow
Установите зависимости и запустите обучение:
```bash
pip install -r requirements.txt
python train.py

### Ответ на Вопрос 47. Поместите ML модель в контейнер

Для контейнеризации ML-модели был написан `Dockerfile`, который собирает образ на базе `python:3.9-slim`, устанавливает зависимости, копирует веса модели (`model.pkl`), scaler (`scaler.pkl`) и код FastAPI-приложения (`app.py`). Сервер запускается через Uvicorn на порту 8000.

**Содержимое Dockerfile (ответ на вопрос 47):**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model.pkl .
COPY scaler.pkl .
COPY app.py .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]```

Сборка выполнена командой docker build -t titanic-ml-api ., запуск — docker run -d -p 8000:8000 --name my-ml-container titanic-ml-api. Модель успешно отвечает на запросы по адресу http://127.0.0.1:8000/docs

### Ответ на Вопрос 48: Код для организации экспериментирования в MLflow
Для отслеживания параметров, метрик и артефактов модели используется библиотека MLflow. Ниже представлен код из скрипта train.py, реализующий логирование эксперимента.
```import mlflow
import mlflow.sklearn

# Настройка имени эксперимента
mlflow.set_experiment("Titanic_Kaggle_MLOps_Experiment")

with mlflow.start_run():
    # 1. Логирование гиперпараметров
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("model_type", "RandomForest")

    # 2. Обучение модели
    # (здесь происходит код обучения модели RandomForestClassifier)
    # model.fit(X_train_scaled, y_train)

    # 3. Логирование метрики
    # acc = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", acc)

    # 4. Сохранение модели в артефакты MLflow
    mlflow.sklearn.log_model(
        model, 
        name="model", 
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )```
