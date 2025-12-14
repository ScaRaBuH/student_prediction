# src/prepare_dataset.py

import pandas as pd
import numpy as np

RESULTS_PATH = "data/raw/результаты.xlsx"
STATUS_PATH = "data/raw/изменение_статуса.xlsx"
OUTPUT_PATH = "data/processed/dataset.csv"

print("Загружаем данные...")

df_results = pd.read_excel(RESULTS_PATH)
df_status = pd.read_excel(STATUS_PATH)

print(f"Успеваемость: {df_results.shape}")
print(f"Статусы: {df_status.shape}")

# Переименуем ИД → PK для удобного merge
df_status = df_status.rename(columns={"ИД": "PK"})

# Определим финальный статус студента
# Сортируем по дате изменения (новые записи в конце) и берём последнюю для каждого студента
df_status_sorted = df_status.sort_values(["PK", "дата изменения"])
final_status = df_status_sorted.groupby("PK").last()[["статус", "выпуск"]].reset_index()

print("\nРаспределение финальных статусов (до маппинга):")
print(final_status["статус"].value_counts())

# Создадим понятную целевую переменную
# Логика на основе твоих данных:
# - Если в последней строке "выпуск" == "выпустился" → 'graduated'
# - Если статус == -1 → 'expelled'
# - Если статус == 3 → предположим 'academic_leave' (надо проверить, что это значит)
# - Если статус == 1 и нет "выпустился" → 'still_studying'
# - Остальное → 'other'

def map_target(row):
    if row["выпуск"] == "выпустился":
        return "graduated"
    elif row["статус"] == -1:
        return "expelled"
    elif row["статус"] == 3:
        return "academic_leave"
    elif row["статус"] == 1:
        return "still_studying"
    else:
        return "other"

final_status["target"] = final_status.apply(map_target, axis=1)

print("\nРаспределение целевой переменной:")
print(final_status["target"].value_counts())

# Теперь выделяем первый семестр
# Для каждого студента берём минимальный SEMESTER
student_first_semester = df_results.groupby("PK")["SEMESTER"].min().reset_index()
student_first_semester = student_first_semester.rename(columns={"SEMESTER": "first_semester"})

# Объединяем с результатами и фильтруем только первый семестр
df_results_with_first = df_results.merge(student_first_semester, on="PK")
df_first_semester = df_results_with_first[
    df_results_with_first["SEMESTER"] == df_results_with_first["first_semester"]
].copy()

print(f"\nОценки за первый семестр: {df_first_semester.shape[0]} записей")

# Создаём признаки по студенту за первый семестр
features = df_first_semester.groupby("PK").agg(
    exams_count=("PK", "size"),
    mean_score=("BALLS", "mean"),
    median_score=("BALLS", "median"),
    max_score=("BALLS", "max"),
    min_score=("BALLS", "min"),
    grade_A_count=("GRADE", lambda x: (x == "A").sum() + (x == "A-").sum() + (x == "A+").sum()),
    grade_B_count=("GRADE", lambda x: (x.str.startswith("B")).sum()),
    grade_C_count=("GRADE", lambda x: (x.str.startswith("C")).sum()),
    grade_D_E_F_count=("GRADE", lambda x: (x.isin(["D", "D+", "E", "Fx", "F"])).sum()),
).reset_index()

print(f"Признаков создано: {features.shape[1]-1} для {features.shape[0]} студентов")

# Объединяем признаки с целевой переменной
dataset = features.merge(final_status[["PK", "target"]], on="PK", how="inner")

print(f"\nФинальный датасет: {dataset.shape}")
print(dataset["target"].value_counts())

# Сохраняем
dataset.to_csv(OUTPUT_PATH, index=False)
print(f"\nДатасет сохранён в {OUTPUT_PATH}")

print("\nГотово!")