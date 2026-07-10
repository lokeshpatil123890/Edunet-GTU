import pandas as pd
import numpy as np
import urllib.request
import os
 
pd.set_option("display.width", 120)
 
base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_URL = "https://raw.githubusercontent.com/datasets/co2-fossil-by-nation/main/data/fossil-fuel-co2-emissions-by-nation.csv"
LOCAL_PATH = os.path.join(base_dir, "fossil-fuel-co2-emissions-by-nation.csv")
 
if not os.path.exists(LOCAL_PATH):
    urllib.request.urlretrieve(DATA_URL, LOCAL_PATH)
 
df = pd.read_csv(LOCAL_PATH)
df.columns = [c.strip() for c in df.columns]
 
print(df.shape)
print(df.groupby("Country").size().sort_values().head())
print(df.isna().sum().sort_values(ascending=False))
 
corr = df.select_dtypes(np.number).corr(numeric_only=True)
print(corr["Total"].sort_values(ascending=False))
print(df.groupby("Year")["Total"].sum().tail(10))
 
df = df.sort_values(["Country", "Year"]).reset_index(drop=True)
 
zero_fill_cols = ["Solid Fuel", "Liquid Fuel", "Gas Fuel", "Cement", "Gas Flaring", "Bunker fuels (Not in Total)"]
for col in zero_fill_cols:
    df[col] = df[col].fillna(0)
 
g = df.groupby("Country")["Per Capita"]
df["Per Capita"] = g.transform(lambda s: s.interpolate(limit_direction="both").fillna(s.median()))
df["Per Capita"] = df["Per Capita"].fillna(df["Per Capita"].median())
 
print(df.isna().sum().sum())
print(df.groupby("Country")["Total"].sum().sort_values(ascending=False).head(5))
 
df.to_csv(os.path.join(base_dir, "fossil_fuel_co2_emissions_cleaned.csv"), index=False)
df.head(10).to_csv(os.path.join(base_dir, "sample_data.csv"), index=False)
df.sort_values("Total", ascending=False).head(10).reset_index(drop=True).to_csv(os.path.join(base_dir, "sorted_data.csv"), index=False)
print("saved cleaned, sample (10 rows), and sorted (10 rows) csv files in the script directory")
 