# src/load_data.py

import os
import pandas as pd

# Пути к файлам
RAW_DIR = "data/raw"

results_path = os.path.join(RAW_DIR, "результаты.xlsx")
status_path = os.path.join(RAW_DIR, "изменение_статуса.xlsx")

# Читаем оба файла
df_results = pd.read_excel(results_path)
df_status = pd.read_excel(status_path)

# Показываем базовую информацию
print("\n=== Результаты успеваемости ===")
print(f"Размер таблицы: {df_results.shape}")
print("Первые 5 строк:")
print(df_results.head())

print("\n=== Изменение статуса ===")
print(f"Размер таблицы: {df_status.shape}")
print("Первые 5 строк:")
print(df_status.head())

# Уникальные значения в столбце статуса (это наш target)
if "статус" in df_status.columns:
    print("\nУникальные значения в столбце 'статус':")
    print(df_status["статус"].value_counts())
else:
    print("\nСтолбец 'статус' не найден. Вот все столбцы:")
    print(df_status.columns.tolist())
