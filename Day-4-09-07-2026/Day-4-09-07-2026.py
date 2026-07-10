import pandas as pd
import numpy as np
import urllib.request
import os
 
pd.set_option("display.width", 120)
 
DATA_URL = "https://raw.githubusercontent.com/AnshTanwar/Global-Data-on-Sustainable-Energy/main/global-data-on-sustainable-energy%20(1).csv"
LOCAL_PATH = "global-data-on-sustainable-energy.csv"
 
if not os.path.exists(LOCAL_PATH):
    urllib.request.urlretrieve(DATA_URL, LOCAL_PATH)
 
df = pd.read_csv(LOCAL_PATH)
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={
    "Density\\n(P/Km2)": "Density(P/Km2)",
    "Access to clean fuels for cooking": "Access_to_clean_fuels_pct",
    "Access to electricity (% of population)": "Access_to_electricity_pct",
})
df["Density(P/Km2)"] = pd.to_numeric(df["Density(P/Km2)"].astype(str).str.replace(",", ""), errors="coerce")
 
print(df.shape)
print(df.groupby("Entity").size().sort_values().head())
print(df.isna().sum().sort_values(ascending=False).head(10))
 
corr = df.select_dtypes(np.number).corr(numeric_only=True)
print(corr["Value_co2_emissions_kt_by_country"].sort_values(ascending=False).head(5))
print(corr["Access_to_electricity_pct"].sort_values(ascending=False).head(5))
 
df = df.sort_values(["Entity", "Year"]).reset_index(drop=True)
df["Financial flows to developing countries (US $)"] = df["Financial flows to developing countries (US $)"].fillna(0)
df = df.drop(columns=["Renewables (% equivalent primary energy)"])
 
fill_cols = [
    "Access_to_electricity_pct", "Access_to_clean_fuels_pct",
    "Renewable-electricity-generating-capacity-per-capita",
    "Renewable energy share in the total final energy consumption (%)",
    "Electricity from fossil fuels (TWh)", "Electricity from nuclear (TWh)",
    "Electricity from renewables (TWh)", "Low-carbon electricity (% electricity)",
    "Energy intensity level of primary energy (MJ/$2017 PPP GDP)",
    "Value_co2_emissions_kt_by_country", "gdp_growth", "gdp_per_capita",
]
for col in fill_cols:
    g = df.groupby("Entity")[col]
    df[col] = g.transform(lambda s: s.interpolate(limit_direction="both").fillna(s.median()))
    df[col] = df[col].fillna(df[col].median())
 
df = df.dropna(subset=["Density(P/Km2)", "Land Area(Km2)", "Latitude", "Longitude"])
print(df.isna().sum().sum())
 
df["decade"] = (df["Year"] // 10) * 10
print(df.groupby("decade")["Access_to_electricity_pct"].mean())
 
df.to_csv("sustainable_energy_cleaned.csv", index=False)
print("saved cleaned csv")