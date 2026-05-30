# preprocessing/population.py

import os
import re
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "R_data", "Population Density")
OUT_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(OUT_DIR, exist_ok=True)

ERP_FILE = os.path.join(
    RAW_DIR,
    "tzp24-population-erp_popd_pnpd-by-tz-2021-2066.csv"
)

AGE_FILE = os.path.join(
    RAW_DIR,
    "tzp24-popd-by-age_sex-by-tz-2021-2066.csv"
)

DICT_FILE = os.path.join(
    RAW_DIR,
    "tzp24-population-data-dictionary.csv"
)

TARGET_YEAR = 2024


def minmax(series):
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    mn = series.min()
    mx = series.max()

    if mx - mn < 1e-9:
        return series * 0

    return (series - mn) / (mx - mn)


def load_population_base():
    print("Reading population base file:")
    print(ERP_FILE)

    df = pd.read_csv(ERP_FILE)

    df = df[df["ROM"] == "Greater Sydney"].copy()

    keep_cols = [
        "TZ_CODE21",
        "TZ_NAME21",
        "SA2_CODE21",
        "SA2_NAME21",
        "SA3_CODE21",
        "SA3_NAME21",
        "SA4_CODE21",
        "SA4_NAME21",
        "LGA_NAME21",
        "GCC Six Cities",
        "ROM",
        f"ERP_{TARGET_YEAR}",
        f"POPD_{TARGET_YEAR}",
        f"PNPD_{TARGET_YEAR}",
        "ERP_2031",
        "ERP_2036",
        "POPD_2031",
        "POPD_2036",
    ]

    df = df[keep_cols].copy()

    df.rename(
        columns={
            f"ERP_{TARGET_YEAR}": "population_2024",
            f"POPD_{TARGET_YEAR}": "population_density_2024",
            f"PNPD_{TARGET_YEAR}": "employment_density_2024",
            "ERP_2031": "population_2031",
            "ERP_2036": "population_2036",
            "POPD_2031": "population_density_2031",
            "POPD_2036": "population_density_2036",
        },
        inplace=True,
    )

    numeric_cols = [
        "population_2024",
        "population_density_2024",
        "employment_density_2024",
        "population_2031",
        "population_2036",
        "population_density_2031",
        "population_density_2036",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["population_growth_2024_2031"] = (
        df["population_2031"] - df["population_2024"]
    )

    df["population_growth_rate_2024_2031"] = (
        df["population_growth_2024_2031"]
        / (df["population_2024"] + 1e-9)
    )

    df["density_growth_2024_2031"] = (
        df["population_density_2031"] - df["population_density_2024"]
    )

    return df


def load_age_summary():
    print("Reading age/sex file:")
    print(AGE_FILE)

    age_df = pd.read_csv(AGE_FILE)
    age_df = age_df[age_df["ROM"] == "Greater Sydney"].copy()

    id_cols = ["TZ_CODE21"]

    age_cols_2024 = [
        col for col in age_df.columns
        if col.endswith(f"_{TARGET_YEAR}")
    ]

    youth_cols = []
    working_cols = []
    senior_cols = []

    for col in age_cols_2024:
        match = re.search(r"_(\d+)-(\d+)_2024", col)

        if match:
            low = int(match.group(1))
            high = int(match.group(2))

            if high <= 19:
                youth_cols.append(col)
            elif low >= 20 and high <= 64:
                working_cols.append(col)
            elif low >= 65:
                senior_cols.append(col)

        elif "100+_2024" in col:
            senior_cols.append(col)

    summary = age_df[id_cols].copy()

    summary["youth_population_2024"] = age_df[youth_cols].sum(axis=1)
    summary["working_age_population_2024"] = age_df[working_cols].sum(axis=1)
    summary["senior_population_2024"] = age_df[senior_cols].sum(axis=1)

    total = (
        summary["youth_population_2024"]
        + summary["working_age_population_2024"]
        + summary["senior_population_2024"]
        + 1e-9
    )

    summary["working_age_ratio_2024"] = (
        summary["working_age_population_2024"] / total
    )

    summary["senior_ratio_2024"] = (
        summary["senior_population_2024"] / total
    )

    return summary


def build_population_dataset():
    print("Building compressed population input...")

    base = load_population_base()
    age = load_age_summary()

    df = base.merge(age, on="TZ_CODE21", how="left")

    fill_cols = [
        "youth_population_2024",
        "working_age_population_2024",
        "senior_population_2024",
        "working_age_ratio_2024",
        "senior_ratio_2024",
    ]

    for col in fill_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # =========================
    # Compact model features
    # =========================
    feature_cols = [
        "population_2024",
        "population_density_2024",
        "employment_density_2024",
        "population_growth_2024_2031",
        "population_growth_rate_2024_2031",
        "density_growth_2024_2031",
        "working_age_ratio_2024",
        "senior_ratio_2024",
    ]

    model_df = df[
        [
            "TZ_CODE21",
            "TZ_NAME21",
            "SA2_CODE21",
            "SA2_NAME21",
            "SA3_CODE21",
            "SA3_NAME21",
            "LGA_NAME21",
            "GCC Six Cities",
            "ROM",
        ]
        + feature_cols
    ].copy()

    # Normalised version for ML
    norm_df = model_df.copy()

    for col in feature_cols:
        norm_df[col + "_norm"] = minmax(norm_df[col])

    # =========================
    # Save outputs
    # =========================

    clean_path = os.path.join(OUT_DIR, "population_cleaned_by_tz.csv")
    model_path = os.path.join(OUT_DIR, "population_features_by_tz.csv")
    norm_path = os.path.join(OUT_DIR, "population_features_by_tz_normalised.csv")

    df.to_csv(clean_path, index=False)
    model_df.to_csv(model_path, index=False)
    norm_df.to_csv(norm_path, index=False)

    print("Saved:", clean_path)
    print("Saved:", model_path)
    print("Saved:", norm_path)

    # Regional summary, useful for report
    lga_summary = model_df.groupby("LGA_NAME21")[feature_cols].sum().reset_index()
    lga_path = os.path.join(OUT_DIR, "population_summary_by_lga.csv")
    lga_summary.to_csv(lga_path, index=False)
    print("Saved:", lga_path)

    print("\nPopulation preprocessing complete.")
    print("Rows:", len(model_df))
    print("Compact feature count:", len(feature_cols))
    print("Features:")
    for col in feature_cols:
        print("-", col)


if __name__ == "__main__":
    build_population_dataset()