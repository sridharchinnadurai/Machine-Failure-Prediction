import pandas as pd

df = pd.read_csv("D:\Idea\predictive_maintenance_balanced.csv")
print(df.info())
df.drop_duplicates(inplace=True)
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        continue

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df.to_csv("../cleaned_machine_failure.csv", index=False)
# D:\Idea\training.py
print("Data cleaning completed")