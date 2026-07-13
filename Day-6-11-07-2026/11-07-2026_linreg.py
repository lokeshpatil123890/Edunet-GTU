import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry_convert as pc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")

raw = pd.read_csv("owid_energy.csv")
cols = ["country", "iso_code", "year", "population", "gdp",
        "energy_per_capita", "electricity_generation", "renewables_share_elec"]
df = raw.loc[
    (raw["year"] == 2022) & (raw["iso_code"].notna()) & (raw["iso_code"].str.len() == 3), cols
].reset_index(drop=True)

print(df.shape)
print(df.isna().sum().sort_values(ascending=False))


def iso3_to_continent(iso3):
    try:
        return pc.country_alpha2_to_continent_code(pc.country_alpha3_to_country_alpha2(iso3))
    except Exception:
        return None


df["continent"] = df["iso_code"].apply(iso3_to_continent)
manual_fix = {"TLS": "AS", "SXM": "NA", "VAT": "EU"}
df["continent"] = df.apply(lambda r: manual_fix.get(r["iso_code"], r["continent"]), axis=1)
df = df[df["continent"].notna()].reset_index(drop=True)
continent_names = {"AF": "Africa", "AS": "Asia", "EU": "Europe",
                    "NA": "North America", "OC": "Oceania", "SA": "South America"}
df["continent"] = df["continent"].map(continent_names)

df = df.dropna(subset=["renewables_share_elec"]).reset_index(drop=True)
for col in ["population", "gdp", "energy_per_capita", "electricity_generation"]:
    df[col] = df.groupby("continent")[col].transform(lambda s: s.fillna(s.median()))
    df[col] = df[col].fillna(df[col].median())

print(df.isna().sum().sum())

df["gdp_per_capita"] = df["gdp"] / df["population"]
df["income_tier"] = pd.qcut(df["gdp_per_capita"].rank(method="first"), q=3, labels=["Low", "Medium", "High"])

numeric_cols = ["population", "gdp", "gdp_per_capita", "energy_per_capita",
                "electricity_generation", "renewables_share_elec"]
corr = df[numeric_cols].corr(numeric_only=True)
print(corr["renewables_share_elec"].sort_values(ascending=False))
print(df.groupby("continent")["renewables_share_elec"].mean().sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["renewables_share_elec"], bins=25, kde=True, ax=ax)
ax.set_title("Distribution of Renewable Electricity Share (2022)")
fig.tight_layout()
fig.savefig("renewables_share_hist.png", dpi=120)

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="gdp_per_capita", y="renewables_share_elec", hue="continent", ax=ax)
ax.set_xscale("log")
ax.set_title("GDP per Capita vs Renewable Electricity Share")
fig.tight_layout()
fig.savefig("gdp_vs_renewables_scatter.png", dpi=120)

fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x="continent", y="renewables_share_elec", ax=ax)
ax.set_title("Renewable Electricity Share by Continent")
fig.tight_layout()
fig.savefig("renewables_by_continent.png", dpi=120)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
ax.set_title("Correlation Heatmap")
fig.tight_layout()
fig.savefig("correlation_heatmap_energy.png", dpi=120)

model_df = df.copy()
scale_cols = ["population", "gdp", "gdp_per_capita", "energy_per_capita", "electricity_generation"]
model_df[scale_cols] = StandardScaler().fit_transform(model_df[scale_cols])
model_df = pd.get_dummies(model_df, columns=["continent"], prefix="continent")

le = LabelEncoder().fit(["Low", "Medium", "High"])
model_df["income_tier_encoded"] = le.transform(df["income_tier"])
model_df = model_df.drop(columns=["income_tier", "country", "iso_code", "year"])

X = model_df.drop(columns=["renewables_share_elec"])
y = model_df["renewables_share_elec"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.3f}")

fig, ax = plt.subplots(figsize=(7, 6))
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.plot([0, 100], [0, 100], "--", color="gray")
ax.set_xlabel("Actual renewables_share_elec")
ax.set_ylabel("Predicted renewables_share_elec")
ax.set_title("Linear Regression: Actual vs Predicted")
fig.tight_layout()
fig.savefig("regression_actual_vs_predicted.png", dpi=120)

df.to_csv("energy_regression_cleaned.csv", index=False)
print("saved cleaned csv + plots")
