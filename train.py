import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn

# --- ЗАДАНИЕ 46: Обучение на датасете Kaggle (Titanic) ---

# 1. Загрузка данных
# Убедитесь, что файл train.csv находится в папке C:\mlops_exam\
df = pd.read_csv('train.csv')

# 2. Простейшая предобработка
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
target = 'Survived'

df = df[features + [target]].copy()

# ИСПРАВЛЕНИЕ PANDAS: убрали inplace=True, так как в новых версиях это вызывает ошибку
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- ЗАДАНИЕ 48: Настройка MLflow ---
mlflow.set_experiment("Titanic_Kaggle_MLOps_Experiment")

with mlflow.start_run():
    # 3. Логирование гиперпараметров
    n_estimators = 100
    max_depth = 5
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("model_type", "RandomForest")

    # 4. Обучение модели
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # 5. Оценка и логирование метрики
    predictions = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", acc)

    # 6. Сохранение модели в MLflow
    # ИСПРАВЛЕНИЕ MLFLOW:
    # - заменили позиционный аргумент на name="model"
    # - добавили skops_trusted_types для обхода ошибки безопасности деревьев
    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    # Сохраняем модель и scaler локально для Docker-контейнера
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print(f"Модель успешно обучена на данных Kaggle! Accuracy: {acc:.4f}")