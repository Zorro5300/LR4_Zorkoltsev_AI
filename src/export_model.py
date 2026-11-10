import json
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import pandas as pd
from sklearn.model_selection import train_test_split

# Указываем локальный сервер MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
c = MlflowClient()

print("Ищем модель по алиасу 'champion'...")
v = c.get_model_version_by_alias("student-risk-classifier", "champion")
print(f"Найдена версия: {v.version}")

# Загружаем модель из MLflow
model = mlflow.sklearn.load_model(f"models:/student-risk-classifier/{v.version}")

# Сохраняем локально в папку exported_model
out = Path("exported_model")
if out.exists():
    shutil.rmtree(out)

mlflow.sklearn.save_model(
    model, 
    str(out), 
    pip_requirements=Path("requirements.txt").read_text().splitlines()
)

# Загружаем данные, чтобы сформировать sample_request
df = pd.read_csv("data/raw/student-mat.csv", sep=";")
df["y"] = (df["G3"] < 10).astype(int)
X = df.drop(columns=["G1", "G2", "G3", "y"])

# Берем 3 объекта для примера
X_test_sample = X.iloc[:3]

# Формируем метаданные
NUMERIC = ["age", "absences", "Medu", "Fedu", "traveltime", "studytime",
           "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health"]
cat_cols = [c for c in X.columns if c not in NUMERIC]

metadata = {
    "name": "student-risk-classifier",
    "version": v.version,
    "run_id": v.run_id,
    "features": X.columns.tolist(),
    "numeric": NUMERIC,
    "categorical": cat_cols,
    "classes": ["no_risk", "risk"],
    "threshold": 10
}
(out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

# Формируем запрос и ожидаемый ответ
request = {"instances": X_test_sample.to_dict(orient="records")}
Path("sample_request.json").write_text(json.dumps(request, indent=2, ensure_ascii=False))
Path("expected_predictions.json").write_text(json.dumps(model.predict(X_test_sample).tolist()))

print("Экспорт завершен успешно!")
print(json.dumps(metadata, indent=2, ensure_ascii=False))