FROM python:3.9-slim

WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем файлы модели, scaler и код приложения
COPY model.pkl .
COPY scaler.pkl .
COPY app.py .

# Открываем порт
EXPOSE 8000

# Команда для запуска сервера
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]