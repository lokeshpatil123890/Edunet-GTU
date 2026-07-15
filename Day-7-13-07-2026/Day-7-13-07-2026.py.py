import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid")

co2 = pd.read_csv("owid_co2.csv")
co2_sub = co2.loc[
    (co2["year"] == 2020) & (co2["iso_code"].notna()) & (co2["iso_code"].str.len() == 3),
    ["country", "iso_code", "population", "gdp", "co2_per_capita", "energy_per_capita"],
]

fa = pd.read_csv("forest_area.csv")
fa_sub = fa.loc[(fa["year"] == 2020) & (fa["code"].notna()), ["code", "forest_area"]]

df = co2_sub.merge(fa_sub, left_on="iso_code", right_on="code", how="inner").drop(columns=["code"])
print(df.shape)
print(df.isna().sum())

df["gdp_per_capita"] = df["gdp"] / df["population"]
df["gdp_per_capita"] = df["gdp_per_capita"].fillna(df["gdp_per_capita"].median())
for col in ["co2_per_capita", "energy_per_capita"]:
    df[col] = df[col].fillna(df[col].median())
print(df.isna().sum().sum())

features = ["co2_per_capita", "energy_per_capita", "gdp_per_capita", "forest_area"]
corr = df[features].corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
ax.set_title("Correlation Heatmap")
fig.tight_layout()
fig.savefig("correlation_heatmap.png", dpi=120)

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["forest_area"], bins=30, kde=True, ax=ax)
ax.set_title("Distribution of Forest Area Share (2020)")
fig.tight_layout()
fig.savefig("forest_area_hist.png", dpi=120)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

inertias, silhouettes = [], []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={silhouettes[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(list(k_range), inertias, marker="o")
axes[0].set_title("Elbow Method")
axes[1].plot(list(k_range), silhouettes, marker="o", color="darkorange")
axes[1].set_title("Silhouette Score by k")
fig.tight_layout()
fig.savefig("elbow_and_silhouette.png", dpi=120)

best_k = k_range[int(np.argmax(silhouettes))]
print("best k:", best_k)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)
print("silhouette:", silhouette_score(X_scaled, df["cluster"]))
print(df["cluster"].value_counts().sort_index())
print(df.groupby("cluster")[features].mean().round(2))

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df["pca1"], df["pca2"] = coords[:, 0], coords[:, 1]

fig, ax = plt.subplots(figsize=(9, 7))
sns.scatterplot(data=df, x="pca1", y="pca2", hue="cluster", palette="tab10", s=70, ax=ax)
ax.set_title(f"K-Means Clusters (k={best_k}) -- PCA 2D Projection")
fig.tight_layout()
fig.savefig("kmeans_clusters_pca.png", dpi=120)

df[["country", "cluster"] + features].round(2).to_csv("forest_co2_clusters_short.csv", index=False)
print("saved short csv + plots")
