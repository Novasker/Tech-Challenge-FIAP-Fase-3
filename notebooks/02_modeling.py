# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "-r ../requirements.txt",
# ]
# ///
# MAGIC %pip install -r ../requirements.txt
# MAGIC # Restart Python so the newly installed packages are picked up
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

# Locate the repository root from the notebook path (this notebook lives in <repo>/notebooks/)
notebook_path = (dbutils.notebook.entry_point.getDbutils()
    .notebook().getContext().notebookPath().get())
repo_root = "/Workspace" + os.path.dirname(os.path.dirname(notebook_path))

# Put the repo first on the path and drop any stale "src" module cached in this session
sys.path.insert(0, repo_root)
for module_name in list(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        del sys.modules[module_name]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.evaluation.validation import cross_validate_grouped, split_by_municipality
from src.modeling.pipelines import build_model_pipeline
from src.preprocessing.build_dataset import load_modeling_sample
from src.preprocessing.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

spark.sql("USE CATALOG workspace")

# Settings used across the notebook
SEED = 42
SAMPLE_FRACTION = 0.15      # share of students loaded into pandas (memory limit of the free workspace)
MODEL_NAMES = ["logistic", "hist_gradient_boosting"]
SHAP_SAMPLE_SIZE = 20000    # rows used for permutation importance and SHAP
PROJECTION_YEARS = 6        # 2024 -> 2030
MIN_STUDENTS = 50           # minimum students per municipality and year to trust its rate
N_CLUSTERS = 4

IMAGES_DIR = os.path.join(repo_root, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# COMMAND ----------

sample_df = load_modeling_sample(spark, sample_fraction=SAMPLE_FRACTION)
print("sample shape:", sample_df.shape)

population_df = spark.table("workspace.bronze.ibge_population").toPandas()
population_df["ano"] = population_df["ano"].astype("int64")

enriched_df = sample_df.merge(population_df, on=["ano", "id_municipio"], how="left")

# The merge must not duplicate students, and every student should have a population value
assert len(enriched_df) == len(sample_df), "population merge duplicated rows"
print("missing population:", enriched_df["populacao"].isna().mean())

# COMMAND ----------

def run_experiment(df, numeric_features, label):
    """Runs grouped cross-validation for every model and prints the metrics."""
    train_part_df, _ = split_by_municipality(df)
    feature_columns = CATEGORICAL_FEATURES + numeric_features

    print(f"===== {label} =====")
    for model_name in MODEL_NAMES:
        pipeline = build_model_pipeline(model_name, numeric_features=numeric_features)
        print(model_name)
        print(cross_validate_grouped(
            pipeline,
            train_part_df[feature_columns],
            train_part_df["target"],
            train_part_df["id_municipio"],
        ).round(3))

# COMMAND ----------

# Model A: baseline
run_experiment(enriched_df, NUMERIC_FEATURES, "Model A (baseline)")

# COMMAND ----------

# Model B: adds the previous-year municipal rate (only available for 2024)
model_b_df = enriched_df[enriched_df["ano"] == 2024].copy()
run_experiment(model_b_df, NUMERIC_FEATURES + ["previous_year_municipal_rate"], "Model B (2024 only)")

# COMMAND ----------

# Model C: adds the municipality population
run_experiment(enriched_df, NUMERIC_FEATURES + ["populacao"], "Model C (population)")

# COMMAND ----------

train_df, test_df = split_by_municipality(enriched_df)
feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES

final_pipeline = build_model_pipeline("logistic")
final_pipeline.fit(train_df[feature_columns], train_df["target"])

# Probability of being literate for every test student
test_proba = final_pipeline.predict_proba(test_df[feature_columns])[:, 1]
test_pred = (test_proba >= 0.5).astype(int)

print("AUC:", round(roc_auc_score(test_df["target"], test_proba), 3))
print("F1:", round(f1_score(test_df["target"], test_pred), 3))
print("accuracy:", round(accuracy_score(test_df["target"], test_pred), 3))
print("baseline (always predict literate):", round(test_df["target"].mean(), 3))
print(confusion_matrix(test_df["target"], test_pred))

# COMMAND ----------

# Out-of-fold probabilities on the training set, grouped by municipality
valid_proba = cross_val_predict(
    build_model_pipeline("logistic"),
    train_df[feature_columns], train_df["target"],
    groups=train_df["id_municipio"],
    cv=GroupKFold(n_splits=5),
    method="predict_proba",
)[:, 1]

thresholds = np.arange(0.40, 0.76, 0.05)
balanced_scores = [
    balanced_accuracy_score(train_df["target"], (valid_proba >= threshold).astype(int))
    for threshold in thresholds
]
best_threshold = thresholds[int(np.argmax(balanced_scores))]
print("best threshold:", round(best_threshold, 2))

# Confirm on the held-out municipalities
test_pred_best = (test_proba >= best_threshold).astype(int)
print("balanced accuracy:", round(balanced_accuracy_score(test_df["target"], test_pred_best), 3))
print("recall (not literate):", round(recall_score(test_df["target"], test_pred_best, pos_label=0), 3))
print("precision (not literate):", round(precision_score(test_df["target"], test_pred_best, pos_label=0), 3))
print(confusion_matrix(test_df["target"], test_pred_best))

# COMMAND ----------

# Permutation importance: how much the AUC drops when one feature is shuffled
shap_sample_df = test_df.sample(n=SHAP_SAMPLE_SIZE, random_state=SEED)

importance = permutation_importance(
    final_pipeline,
    shap_sample_df[feature_columns], shap_sample_df["target"],
    scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1,
)

importance_df = (pd.DataFrame({
        "feature": feature_columns,
        "auc_drop": importance.importances_mean,
    })
    .sort_values("auc_drop", ascending=False))
print(importance_df)

# COMMAND ----------

# SHAP values for the linear model, summed back from one-hot columns to the original feature
preprocessor = final_pipeline.named_steps["preprocessor"]
model = final_pipeline.named_steps["model"]

X_shap = preprocessor.transform(shap_sample_df[feature_columns])
transformed_names = preprocessor.get_feature_names_out()

explainer = shap.LinearExplainer(model, X_shap[:2000])
shap_values = explainer.shap_values(X_shap)

shap_by_feature = {}
for feature in feature_columns:
    column_indexes = [
        index for index, name in enumerate(transformed_names)
        if name.startswith(f"numeric__{feature}") or name.startswith(f"categorical__{feature}_")
    ]
    shap_by_feature[feature] = shap_values[:, column_indexes].sum(axis=1)

shap_importance_df = (pd.DataFrame(
        {feature: np.abs(values).mean() for feature, values in shap_by_feature.items()},
        index=["mean_abs_shap"])
    .T.sort_values("mean_abs_shap", ascending=False))
print(shap_importance_df)

# Save the chart for the README
shap_importance_df["mean_abs_shap"].sort_values().plot(kind="barh", figsize=(7, 4))
plt.title("Mean |SHAP| by feature")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "shap_importance.png"), dpi=150)
plt.show()

# COMMAND ----------

# Municipal network only, since municipal goals exist only for it
indicator_df = spark.table("workspace.gold.indicador_alfabetizacao_municipio").toPandas()
municipal_df = indicator_df[indicator_df["rede"] == "Municipal"]

# One row per municipality with the literacy rate of each year
rates_df = municipal_df.pivot_table(index="id_municipio", columns="ano", values="taxa_alfabetizacao")
rates_df.columns = [f"rate_{year}" for year in rates_df.columns]

municipality_info_df = municipal_df[municipal_df["ano"] == 2024][
    ["id_municipio", "municipio_nome", "nome_uf", "nome_regiao", "amazonia_legal", "meta_alfabetizacao_2030"]
]
risk_df = municipality_info_df.merge(rates_df.reset_index(), on="id_municipio")

# Number of students per municipality and year, used to flag unreliable rates
students_count_df = (spark.table("workspace.gold.base_modelagem_alunos")
    .filter("rede = 'Municipal'")
    .groupBy("id_municipio", "ano").count().toPandas()
    .pivot(index="id_municipio", columns="ano", values="count"))
students_count_df.columns = [f"students_{year}" for year in students_count_df.columns]
risk_df = risk_df.merge(students_count_df.reset_index(), on="id_municipio", how="left")

risk_df["reliable"] = (risk_df["students_2023"] >= MIN_STUDENTS) & (risk_df["students_2024"] >= MIN_STUDENTS)

# Current distance to the goal (more stable) and the simple trend projection (kept within 0-100)
risk_df["yearly_change"] = risk_df["rate_2024"] - risk_df["rate_2023"]
risk_df["current_gap"] = risk_df["meta_alfabetizacao_2030"] - risk_df["rate_2024"]
risk_df["projected_2030"] = (risk_df["rate_2024"] + PROJECTION_YEARS * risk_df["yearly_change"]).clip(0, 100)
risk_df["projected_gap"] = risk_df["meta_alfabetizacao_2030"] - risk_df["projected_2030"]
risk_df["will_miss_goal"] = risk_df["projected_2030"] < risk_df["meta_alfabetizacao_2030"]

reliable_df = risk_df[risk_df["reliable"]].dropna(
    subset=["meta_alfabetizacao_2030", "rate_2024", "yearly_change"])

print("reliable municipalities:", len(reliable_df))
print("share projected to miss the goal:", round(reliable_df["will_miss_goal"].mean(), 3))
print(reliable_df.groupby("nome_regiao")["will_miss_goal"].mean().round(3))

# Ranking by current distance to the goal
print(reliable_df.sort_values("current_gap", ascending=False).head(15)[
    ["municipio_nome", "nome_uf", "students_2024", "rate_2024", "current_gap"]
].to_string())

spark.createDataFrame(risk_df).write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable("workspace.gold.risco_municipios")

# COMMAND ----------

cluster_features = ["rate_2024", "yearly_change", "current_gap"]
cluster_df = reliable_df.dropna(subset=cluster_features).copy()

# Scale the features so no single one dominates the distance
scaled_features = StandardScaler().fit_transform(cluster_df[cluster_features])
cluster_df["cluster"] = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10).fit_predict(scaled_features)

# Profile of each cluster, its size and how each region is distributed across clusters
print(cluster_df.groupby("cluster")[cluster_features + ["students_2024"]].mean().round(1))
print(cluster_df["cluster"].value_counts().sort_index())
print(cluster_df.groupby("nome_regiao")["cluster"].value_counts(normalize=True).unstack().round(2))

# Save the chart for the README
for cluster_id, group_df in cluster_df.groupby("cluster"):
    plt.scatter(group_df["rate_2024"], group_df["yearly_change"], s=6, label=f"cluster {cluster_id}")
plt.xlabel("literacy rate 2024")
plt.ylabel("yearly change")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "clusters.png"), dpi=150)
plt.show()