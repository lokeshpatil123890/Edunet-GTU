import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

sns.set_theme(style="whitegrid")

file_name = "sustainability_jobs.csv"
file_path = file_name

if not os.path.exists(file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)
    if not os.path.exists(file_path):
        file_path = os.path.join(os.path.dirname(script_dir), file_name)
        if not os.path.exists(file_path):
            file_path = os.path.join(os.path.dirname(os.path.dirname(script_dir)), file_name)

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find {file_name} in current, script, parent, or grandparent directories.")

fixed_lines = []
with open(file_path, "r", encoding="utf-8") as f:
    header = f.readline().strip()
    fixed_lines.append(header)
    job_id_counter = 1
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if parts[0].isdigit():
            job_id_counter = int(parts[0]) + 1
            fixed_lines.append(line)
        else:
            fixed_line = f"{job_id_counter},{line}"
            job_id_counter += 1
            fixed_lines.append(fixed_line)

df = pd.read_csv(io.StringIO("\n".join(fixed_lines)))

string_cols = ["job_title", "industry", "country", "company_size", "education_level", "green_certification"]
for col in string_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

print(df.shape)
print(df.isna().sum().sort_values(ascending=False))

numeric_cols = ["experience_years", "remote_ratio", "sustainability_score", "carbon_offset_days", "salary_usd"]
print(df[numeric_cols].corr()["salary_usd"].sort_values(ascending=False))

df["green_certification"] = df["green_certification"].fillna(df["green_certification"].mode()[0] if not df["green_certification"].mode().empty else "No")
df["education_level"] = df["education_level"].fillna(df["education_level"].mode()[0] if not df["education_level"].mode().empty else "Bachelors")


for col in ["experience_years", "remote_ratio", "salary_usd"]:
    df[col] = df.groupby("job_title")[col].transform(lambda s: s.fillna(s.median())).fillna(df[col].median())
for col in ["sustainability_score", "carbon_offset_days"]:
    df[col] = df.groupby("green_certification")[col].transform(lambda s: s.fillna(s.median())).fillna(df[col].median())

print(df.isna().sum().sum())

df["experience_band"] = pd.cut(df["experience_years"], bins=[-1, 2, 5, 10, 20, 100], labels=["0-2", "3-5", "6-10", "11-20", "20+"])
df["salary_per_experience_year"] = df["salary_usd"] / (df["experience_years"] + 1)
df["green_engagement_score"] = df["green_certification"].map({"Yes": 50, "No": 0}) + df["carbon_offset_days"].clip(upper=50)

script_dir = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="experience_years", y="salary_usd", hue="education_level", ax=ax)
ax.set_title("Salary vs Years of Experience")
fig.tight_layout()
fig.savefig(os.path.join(script_dir, "salary_by_experience.png"), dpi=120)

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="carbon_offset_days", y="salary_usd", hue="green_certification", ax=ax)
ax.set_title("Salary vs Carbon Offset Days")
fig.tight_layout()
fig.savefig(os.path.join(script_dir, "salary_vs_offset.png"), dpi=120)

model_df = df.copy()
scale_cols = numeric_cols + ["salary_per_experience_year", "green_engagement_score"]
model_df[scale_cols] = StandardScaler().fit_transform(model_df[scale_cols])

nominal_cols = ["job_title", "industry", "country", "company_size", "green_certification"]
model_df = pd.get_dummies(model_df, columns=nominal_cols, prefix=nominal_cols)

le_edu = LabelEncoder().fit(["Diploma", "Bachelors", "Masters", "PhD"])
le_band = LabelEncoder().fit(["0-2", "3-5", "6-10", "11-20", "20+"])
model_df["education_level_encoded"] = le_edu.transform(df["education_level"])
model_df["experience_band_encoded"] = le_band.transform(df["experience_band"].astype(str))
model_df = model_df.drop(columns=["education_level", "experience_band"])

df.to_csv(os.path.join(script_dir, "sustainability_jobs_processed.csv"), index=False)
model_df.to_csv(os.path.join(script_dir, "sustainability_jobs.csv"), index=False)
print("saved processed and model-ready csv files + 2 plots")