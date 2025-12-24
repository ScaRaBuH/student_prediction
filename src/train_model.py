# src/train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import json
import os

DATASET_PATH = "data/processed/dataset.csv"
MODEL_PATH = "models/model.joblib"
METRICS_PATH = "metrics/metrics.json"

os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

df = pd.read_csv(DATASET_PATH)

print(f"Размер датасета: {df.shape}")
print("Распределение классов:")
print(df["target"].value_counts())

# Удаляем редкий класс
df = df[df["target"] != "still_studying"].copy()

X = df.drop(columns=["PK", "target"])
y = df["target"]

# Кодируем целевую переменную
label_map = {"graduated": 0, "expelled": 1, "academic_leave": 2}
y_encoded = y.map(label_map)

print(f"Пропуски в признаках до обработки:\n{X.isna().sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Обучаем модели на {X_train.shape[0]} примерах...")

# RandomForest с балансировкой
rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # заполняем медианой
    ("model", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ))
])

# LogisticRegression с балансировкой
lr_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])

# Обучаем обе
rf_pipeline.fit(X_train, y_train)
lr_pipeline.fit(X_train, y_train)

rf_pred = rf_pipeline.predict(X_test)
lr_pred = lr_pipeline.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)

# Выбираем лучшую
if rf_accuracy >= lr_accuracy:
    best_pipeline = rf_pipeline
    best_pred = rf_pred
    best_accuracy = rf_accuracy
    model_name = "RandomForest (balanced, n=300)"
else:
    best_pipeline = lr_pipeline
    best_pred = lr_pred
    best_accuracy = lr_accuracy
    model_name = "LogisticRegression (balanced)"

report = classification_report(
    y_test, best_pred,
    output_dict=True,
    target_names=["graduated", "expelled", "academic_leave"]
)

print(f"\nЛучшая модель: {model_name}")
print(f"Accuracy: {best_accuracy:.3f}")
print("Classification report:")
print(classification_report(
    y_test, best_pred,
    target_names=["graduated", "expelled", "academic_leave"]
))

# Сохраняем лучшую модель
joblib.dump(best_pipeline, MODEL_PATH)
print(f"Модель сохранена в {MODEL_PATH}")

# Сохраняем метрики
metrics = {
    "accuracy": best_accuracy,
    "model": model_name,
    "classification_report": report
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

print(f"Метрики сохранены в {METRICS_PATH}")
