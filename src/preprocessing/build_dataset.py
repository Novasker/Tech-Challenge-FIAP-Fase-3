"""Builds the pandas modelling dataset from the gold student table."""
from pyspark.sql import functions as F

BASE_TABLE = "workspace.gold.base_modelagem_alunos"


def load_modeling_sample(spark, sample_fraction=0.15, seed=42):
    """Reads the gold table, adds size features and returns a pandas sample.

    The size features are plain counts (no target involved), computed on the
    full data before sampling so they are not distorted by the sample.
    """
    base_df = spark.table(BASE_TABLE)

    school_size_df = (base_df.groupBy("ano", "id_escola")
        .agg(F.count("*").alias("school_size")))
    municipality_size_df = (base_df.groupBy("ano", "id_municipio")
        .agg(F.count("*").alias("municipality_size")))

    enriched_df = (base_df
        .join(school_size_df, on=["ano", "id_escola"], how="left")
        .join(municipality_size_df, on=["ano", "id_municipio"], how="left")
        .withColumn("target", F.col("alfabetizado").cast("int"))
        .withColumn("amazonia_legal", F.col("amazonia_legal").cast("string")))

    return enriched_df.sample(fraction=sample_fraction, seed=seed).toPandas()
