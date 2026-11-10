import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from mlflow.models import infer_signature
import mlflow, mlflow.sklearn

THRESHOLD = 10
DROP_COLS = ["G1", "G2", "G3"]          # исключаем утечку
NUMERIC = ["age","absences","Medu","Fedu","traveltime","studytime",
           "failures","famrel","freetime","goout","Dalc","Walc","health"]

def load_data():
    df = pd.read_csv("data/raw/student-mat.csv", sep=";")
    df["y"] = (df["G3"] < THRESHOLD).astype(int)
    X = df.drop(columns=DROP_COLS + ["y"])
    y = df["y"]
    return X, y, df

def build_pipeline(algorithm, params, cat_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    clf = (LogisticRegression(**params, max_iter=2000, random_state=42)
           if algorithm == "lr"
           else RandomForestClassifier(**params, random_state=42, n_jobs=1))
    return Pipeline([("pre", pre), ("clf", clf)])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", choices=["lr", "rf"], default="lr")
    p.add_argument("--c", type=float, default=1.0)              # для lr
    p.add_argument("--trees", type=int, default=100)            # для rf
    p.add_argument("--depth", type=int, default=5)              # для rf
    p.add_argument("--register", action="store_true")
    a = p.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("lab4-student-risk")

    X, y, df = load_data()
    cat_cols = [c for c in X.columns if c not in NUMERIC]

    # 60/20/20 + stratify
    Xdev, Xtest, ydev, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    Xtrain, Xval, ytrain, yval = train_test_split(
        Xdev, ydev, test_size=0.25, random_state=42, stratify=ydev)

    params = ({"C": a.c} if a.algorithm == "lr"
              else {"n_estimators": a.trees, "max_depth": a.depth})
    model = build_pipeline(a.algorithm, params, cat_cols)

    digest = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()

    with mlflow.start_run(run_name=f"{a.algorithm}-risk") as run:
        mlflow.log_params({
            "algorithm": a.algorithm, "seed": 42,
            "dataset": "UCI student-mat (dataset 320)",
            "target": f"G3 < {THRESHOLD}", "threshold": THRESHOLD,
            "n_train": len(Xtrain), "n_validation": len(Xval), "n_test": len(Xtest),
            **params,
        })
        mlflow.set_tags({
            "dataset_sha256": digest,
            "purpose": "lab4-var9",
            "selection_metric": "val_f1_macro",
            "note": "excluded G1,G2,G3 to avoid leakage",
        })

        model.fit(Xtrain, ytrain)
        pred = model.predict(Xval)

        mlflow.log_metrics({
            "val_accuracy": accuracy_score(yval, pred),
            "val_f1_macro": f1_score(yval, pred, average="macro"),
        })

        # сплиты и исходные данные
        mlflow.log_dict({
            "train_indices": Xtrain.index.tolist(),
            "validation_indices": Xval.index.tolist(),
            "test_indices": Xtest.index.tolist(),
            "columns": X.columns.tolist(),
            "numeric": NUMERIC, "categorical": cat_cols,
            "threshold": THRESHOLD,
        }, "split.json")
        mlflow.log_artifact("data/raw/student-mat.csv", artifact_path="data")
        mlflow.log_artifact("data/raw/student-mat.sha256", artifact_path="data")
        mlflow.log_artifact("requirements.txt", artifact_path="source")
        mlflow.log_artifact(__file__, artifact_path="source")

        fig, ax = plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay.from_predictions(
            yval, pred, labels=[0,1],
            display_labels=["no_risk","risk"], ax=ax)
        mlflow.log_figure(fig, "validation_confusion_matrix.png")
        plt.close(fig)

        info = mlflow.sklearn.log_model(
            model, name="model",
            input_example=Xtrain.head(2),
            signature=infer_signature(Xtrain, model.predict(Xtrain)),
            pip_requirements=Path("requirements.txt").read_text().splitlines(),
        )
        print(json.dumps({"run_id": run.info.run_id,
                          "model_uri": info.model_uri}, indent=2))

        if a.register:
            mv = mlflow.register_model(info.model_uri, "student-risk-classifier")
            print("REGISTERED_VERSION=", mv.version)

if __name__ == "__main__":
    main()