import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "Pclass": 3,
    "Age": 22.0,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25
}

print("Отправляем запрос в контейнер...")
try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Успех! Ответ от модели:", response.json())
    else:
        print("Ошибка сервера:", response.text)
except Exception as e:
    print("Ошибка подключения:", e)