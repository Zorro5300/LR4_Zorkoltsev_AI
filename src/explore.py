# src/explore.py
import pandas as pd

df = pd.read_csv("data/raw/student-mat.csv", sep=";")
print("shape:", df.shape)
print("\ncolumns:\n", list(df.columns))
print("\nG3 describe:\n", df["G3"].describe())
print("\nclass balance (G3 < 10):",
      (df["G3"] < 10).mean())