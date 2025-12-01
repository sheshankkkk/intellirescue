import pandas as pd

df = pd.read_csv("data/disasters.csv")

print("=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== INFO ===")
print(df.info())
