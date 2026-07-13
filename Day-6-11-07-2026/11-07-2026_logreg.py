import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)

sns.set_theme(style="whitegrid")

df = pd.read_csv("water_potability.csv")
print(df.shape)
print(df["Potability"].value_counts(normalize=True).round(3))
print(df.isna().sum().sort_values(ascending=False))

for col in ["ph", "Sulfate", "Trihalomethanes"]:
    df[col] = df[col].fillna(df[col].median())
print(df.isna().sum().sum())

df["ph_category"] = pd.cut(df["ph"], bins=[-0.1, 6.5, 7.5, 14], labels=["Acidic", "Neutral", "Alkaline"])

contam_cols = ["Chloramines", "Organic_carbon", "Trihalomethanes", "Turbidity"]
z = (df[contam_cols] - df[contam_cols].mean()) / df[contam_cols].std()
df["dominant_contaminant"] = z.idxmax(axis=1)
print(df["dominant_contaminant"].value_counts())

numeric_cols = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
corr = df[numeric_cols + ["Potability"]].corr(numeric_only=True)
print(corr["Potability"].sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(6, 5))
sns.countplot(data=df, x="Potability", ax=ax)
ax.set_title("Potability Class Balance (0 = Not Potable, 1 = Potable)")
fig.tight_layout()
fig.savefig("potability_balance.png", dpi=120)

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=df, x="ph", hue="Potability", bins=30, kde=True, ax=ax)
ax.set_title("pH Distribution by Potability")
fig.tight_layout()
fig.savefig("ph_distribution.png", dpi=120)

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Potability", y="Sulfate", ax=ax)
ax.set_title("Sulfate by Potability")
fig.tight_layout()
fig.savefig("sulfate_by_potability.png", dpi=120)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
ax.set_title("Correlation Heatmap")
fig.tight_layout()
fig.savefig("correlation_heatmap_water.png", dpi=120)

model_df = df.copy()
model_df[numeric_cols] = StandardScaler().fit_transform(model_df[numeric_cols])
model_df = pd.get_dummies(model_df, columns=["dominant_contaminant"], prefix="dominant")

le = LabelEncoder().fit(["Acidic", "Neutral", "Alkaline"])
model_df["ph_category_encoded"] = le.transform(df["ph_category"])
model_df = model_df.drop(columns=["ph_category"])

X = model_df.drop(columns=["Potability"])
y = model_df["Potability"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
print(f"Accuracy  : {acc:.3f}")
print(f"Precision : {prec:.3f}")
print(f"Recall    : {rec:.3f}")
print(f"F1-score  : {f1:.3f}")
print(f"ROC-AUC   : {auc:.3f}")
print(cm)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Potable", "Potable"], yticklabels=["Not Potable", "Potable"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Logistic Regression: Confusion Matrix")
fig.tight_layout()
fig.savefig("confusion_matrix.png", dpi=120)

fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
fig.tight_layout()
fig.savefig("roc_curve.png", dpi=120)

df.to_csv("water_potability_cleaned.csv", index=False)
print("saved cleaned csv + plots")
