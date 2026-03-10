"""
Traffic Accident Data Analysis System
Core analysis module: data loading, cleaning, EDA, feature engineering,
geospatial analysis, ML modeling (XGBoost), and SHAP explainability.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from sklearn.cluster import DBSCAN
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
os.makedirs(VISUALS_DIR, exist_ok=True)

# Resolve dataset path: prefer local copy, fall back to kagglehub cache
LOCAL_CSV = os.path.join(DATA_DIR, "US_Accidents_March23.csv")
KAGGLE_CACHE = os.path.join(
    os.path.expanduser("~"),
    ".cache", "kagglehub", "datasets",
    "sobhanmoosavi", "us-accidents", "versions", "13",
    "US_Accidents_March23.csv",
)
DATASET_PATH = LOCAL_CSV if os.path.exists(LOCAL_CSV) else KAGGLE_CACHE


# ===================================================================
# 1. DATA LOADING
# ===================================================================
def load_data(nrows=None):
    """Load the US-Accidents dataset and return a DataFrame."""
    print(f"Loading data from {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH, nrows=nrows)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} cols")
    return df


# ===================================================================
# 2. DATA CLEANING
# ===================================================================
def clean_data(df):
    """Clean: parse dates, drop unused cols, handle missing values."""
    # Parse datetime columns
    for col in ["Start_Time", "End_Time", "Weather_Timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Drop columns with very high missing-ness or low utility
    drop_cols = [
        "End_Lat", "End_Lng", "ID", "Description",
        "Wind_Chill(F)", "Precipitation(in)", "Weather_Timestamp",
        "Turning_Loop", "Airport_Code", "Zipcode", "Country",
        "Number",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns],
            inplace=True, errors="ignore")

    # Fill numeric NaN with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical NaN with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown")

    # Remove full duplicates
    df.drop_duplicates(inplace=True)

    print(f"After cleaning: {len(df):,} rows x {df.shape[1]} cols")
    return df


# ===================================================================
# 3. FEATURE ENGINEERING
# ===================================================================
def engineer_features(df):
    """Extract temporal and derived features."""
    df["Hour"] = df["Start_Time"].dt.hour
    df["DayOfWeek"] = df["Start_Time"].dt.day_name()
    df["Month"] = df["Start_Time"].dt.month
    df["Year"] = df["Start_Time"].dt.year
    df["Is_Weekend"] = df["DayOfWeek"].isin(["Saturday", "Sunday"]).astype(int)

    # Combined Place column
    df["Place"] = (
        df["Street"].fillna("") + ", " +
        df["City"].fillna("") + ", " +
        df["State"].fillna("")
    )

    # Duration in minutes
    if "End_Time" in df.columns:
        df["Duration_min"] = (
            (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
        ).clip(lower=0)
    return df


# ===================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ===================================================================
def eda_severity_distribution(df):
    """Severity distribution bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    df["Severity"].value_counts().sort_index().plot(kind="bar", color="coral", ax=ax)
    ax.set_title("Accident Severity Distribution", fontsize=14)
    ax.set_xlabel("Severity Level")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "severity_distribution.png"), dpi=150)
    plt.close()
    print("  ✓ severity_distribution.png saved")


def eda_accidents_by_hour(df):
    """Accidents by hour line/bar plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    df["Hour"].value_counts().sort_index().plot(kind="line", marker="o", ax=ax,
                                                color="steelblue")
    ax.set_title("Accidents by Hour of Day", fontsize=14)
    ax.set_xlabel("Hour (24h)")
    ax.set_ylabel("Number of Accidents")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "accidents_by_hour.png"), dpi=150)
    plt.close()
    print("  ✓ accidents_by_hour.png saved")


def eda_accidents_by_day(df):
    """Accidents by day-of-week bar chart."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(9, 5))
    day_counts = df["DayOfWeek"].value_counts().reindex(order)
    day_counts.plot(kind="bar", color="mediumseagreen", ax=ax)
    ax.set_title("Accidents by Day of Week", fontsize=14)
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "accidents_by_day.png"), dpi=150)
    plt.close()
    print("  ✓ accidents_by_day.png saved")


def eda_monthly_trend(df):
    """Monthly accident trend."""
    fig, ax = plt.subplots(figsize=(10, 5))
    df.groupby("Month").size().plot(kind="bar", color="darkorange", ax=ax)
    ax.set_title("Monthly Accident Count", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "monthly_trend.png"), dpi=150)
    plt.close()
    print("  ✓ monthly_trend.png saved")


def eda_top_states(df, n=15):
    """Top N states by accident count."""
    fig, ax = plt.subplots(figsize=(10, 5))
    df["State"].value_counts().head(n).plot(kind="barh", color="slateblue", ax=ax)
    ax.set_title(f"Top {n} States by Accident Count", fontsize=14)
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "top_states.png"), dpi=150)
    plt.close()
    print("  ✓ top_states.png saved")


def eda_top_cities(df, n=15):
    """Top N cities by accident count."""
    fig, ax = plt.subplots(figsize=(10, 5))
    df["City"].value_counts().head(n).plot(kind="barh", color="crimson", ax=ax)
    ax.set_title(f"Top {n} Cities by Accident Count", fontsize=14)
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "top_cities.png"), dpi=150)
    plt.close()
    print("  ✓ top_cities.png saved")


def eda_weather_impact(df, n=15):
    """Top N weather conditions during accidents."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df["Weather_Condition"].value_counts().head(n).plot(kind="barh",
                                                        color="teal", ax=ax)
    ax.set_title(f"Top {n} Weather Conditions During Accidents", fontsize=14)
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "weather_impact.png"), dpi=150)
    plt.close()
    print("  ✓ weather_impact.png saved")


def eda_temperature_distribution(df):
    """Temperature distribution for accidents."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["Temperature(F)"].dropna(), bins=60, color="salmon", edgecolor="k",
            alpha=0.7)
    ax.set_title("Temperature Distribution During Accidents", fontsize=14)
    ax.set_xlabel("Temperature (°F)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "temperature_dist.png"), dpi=150)
    plt.close()
    print("  ✓ temperature_dist.png saved")


def eda_severity_heatmap(df):
    """Heatmap: Hour vs DayOfWeek accident counts."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
    pivot = df.pivot_table(index="DayOfWeek", columns="Hour",
                           values="Severity", aggfunc="count")
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3, ax=ax)
    ax.set_title("Accidents Heatmap: Day of Week vs Hour", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "severity_heatmap.png"), dpi=150)
    plt.close()
    print("  ✓ severity_heatmap.png saved")


def eda_sunrise_sunset(df):
    """Day vs Night accident comparison."""
    if "Sunrise_Sunset" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    df["Sunrise_Sunset"].value_counts().plot(kind="bar", color=["gold", "midnightblue"],
                                              ax=ax)
    ax.set_title("Day vs Night Accidents", fontsize=14)
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "day_vs_night.png"), dpi=150)
    plt.close()
    print("  ✓ day_vs_night.png saved")


def generate_combined_plots(df):
    """Render all key EDA charts into a single combined plots.png image."""
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday"]

    fig, axes = plt.subplots(5, 2, figsize=(20, 28))
    fig.suptitle("Traffic Accident Data Analysis \u2014 Summary",
                 fontsize=22, fontweight="bold", y=0.995)

    # 1 – Severity Distribution
    ax = axes[0, 0]
    df["Severity"].value_counts().sort_index().plot(kind="bar", color="coral", ax=ax)
    ax.set_title("Accident Severity Distribution")
    ax.set_xlabel("Severity Level")
    ax.set_ylabel("Count")

    # 2 – Accidents by Hour
    ax = axes[0, 1]
    df["Hour"].value_counts().sort_index().plot(kind="line", marker="o",
                                                 color="steelblue", ax=ax)
    ax.set_title("Accidents by Hour of Day")
    ax.set_xlabel("Hour (24h)")
    ax.set_ylabel("Number of Accidents")
    ax.set_xticks(range(0, 24))

    # 3 – Accidents by Day of Week
    ax = axes[1, 0]
    df["DayOfWeek"].value_counts().reindex(day_order).plot(kind="bar",
                                                            color="mediumseagreen", ax=ax)
    ax.set_title("Accidents by Day of Week")
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")

    # 4 – Monthly Trend
    ax = axes[1, 1]
    df.groupby("Month").size().plot(kind="bar", color="darkorange", ax=ax)
    ax.set_title("Monthly Accident Count")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")

    # 5 – Top 10 States
    ax = axes[2, 0]
    df["State"].value_counts().head(10).plot(kind="barh", color="slateblue", ax=ax)
    ax.set_title("Top 10 States by Accident Count")
    ax.set_xlabel("Count")
    ax.invert_yaxis()

    # 6 – Top 10 Cities
    ax = axes[2, 1]
    df["City"].value_counts().head(10).plot(kind="barh", color="crimson", ax=ax)
    ax.set_title("Top 10 Cities by Accident Count")
    ax.set_xlabel("Count")
    ax.invert_yaxis()

    # 7 – Top 10 Weather Conditions
    ax = axes[3, 0]
    df["Weather_Condition"].value_counts().head(10).plot(kind="barh",
                                                          color="teal", ax=ax)
    ax.set_title("Top 10 Weather Conditions During Accidents")
    ax.set_xlabel("Count")
    ax.invert_yaxis()

    # 8 – Temperature Distribution
    ax = axes[3, 1]
    ax.hist(df["Temperature(F)"].dropna(), bins=60, color="salmon",
            edgecolor="k", alpha=0.7)
    ax.set_title("Temperature Distribution During Accidents")
    ax.set_xlabel("Temperature (°F)")
    ax.set_ylabel("Frequency")

    # 9 – Severity Heatmap (Hour × Day of Week)
    ax = axes[4, 0]
    pivot = df.pivot_table(index="DayOfWeek", columns="Hour",
                           values="Severity", aggfunc="count")
    pivot = pivot.reindex(day_order)
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Accidents Heatmap: Day of Week vs Hour")

    # 10 – Day vs Night
    ax = axes[4, 1]
    if "Sunrise_Sunset" in df.columns:
        df["Sunrise_Sunset"].value_counts().plot(
            kind="bar", color=["gold", "midnightblue"], ax=ax)
        ax.set_title("Day vs Night Accidents")
        ax.set_ylabel("Count")
    else:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(VISUALS_DIR, "plots.png"), dpi=200)
    plt.close()
    print("  ✓ plots.png saved (combined summary)")


def run_all_eda(df):
    """Execute all EDA plots."""
    print("\n=== Running EDA Visualizations ===")
    eda_severity_distribution(df)
    eda_accidents_by_hour(df)
    eda_accidents_by_day(df)
    eda_monthly_trend(df)
    eda_top_states(df)
    eda_top_cities(df)
    eda_weather_impact(df)
    eda_temperature_distribution(df)
    eda_severity_heatmap(df)
    eda_sunrise_sunset(df)
    generate_combined_plots(df)
    print("=== EDA Complete ===\n")


# ===================================================================
# 5. GEOSPATIAL ANALYSIS
# ===================================================================
def create_heatmap(df, sample_n=50_000):
    """Create Folium heatmap of accident locations."""
    print("Creating geospatial heatmap ...")
    sample = df.dropna(subset=["Start_Lat", "Start_Lng"]).sample(
        n=min(sample_n, len(df)), random_state=42
    )
    m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB dark_matter")
    heat_data = sample[["Start_Lat", "Start_Lng"]].values.tolist()
    HeatMap(heat_data, radius=6, blur=10, max_zoom=13).add_to(m)
    out = os.path.join(VISUALS_DIR, "accident_heatmap.html")
    m.save(out)
    print(f"  ✓ Heatmap saved → {out}")
    return m


def dbscan_hotspots(df, eps_km=1.5, min_samples=50, sample_n=100_000):
    """DBSCAN clustering to identify geographic hotspots."""
    print("Running DBSCAN hotspot detection ...")
    sample = df.dropna(subset=["Start_Lat", "Start_Lng"]).sample(
        n=min(sample_n, len(df)), random_state=42
    )
    coords = sample[["Start_Lat", "Start_Lng"]].values
    # eps in radians: km / earth_radius
    eps_rad = eps_km / 6371.0
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine",
                algorithm="ball_tree")
    labels = db.fit_predict(np.radians(coords))
    sample = sample.copy()
    sample["Cluster"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  ✓ DBSCAN found {n_clusters} hotspot clusters")

    # Plot clusters on Folium map
    m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB positron")
    clustered = sample[sample["Cluster"] != -1]
    for _, row in clustered.iterrows():
        folium.CircleMarker(
            location=[row["Start_Lat"], row["Start_Lng"]],
            radius=2, color="red", fill=True, fill_opacity=0.5
        ).add_to(m)
    out = os.path.join(VISUALS_DIR, "dbscan_hotspots.html")
    m.save(out)
    print(f"  ✓ DBSCAN hotspot map saved → {out}")
    return sample, n_clusters


# ===================================================================
# 6. MACHINE LEARNING — XGBoost Severity Prediction
# ===================================================================
def prepare_ml_data(df):
    """Prepare features and labels for ML training."""
    feature_cols = [
        "Hour", "Month", "Is_Weekend",
        "Temperature(F)", "Humidity(%)", "Pressure(in)",
        "Visibility(mi)", "Wind_Speed(mph)", "Distance(mi)",
    ]
    bool_cols = [
        "Amenity", "Bump", "Crossing", "Give_Way", "Junction",
        "No_Exit", "Railway", "Roundabout", "Station", "Stop",
        "Traffic_Calming", "Traffic_Signal",
    ]
    cat_cols_to_encode = ["Sunrise_Sunset"]

    # Keep only rows with complete data on selected features
    all_cols = feature_cols + bool_cols + cat_cols_to_encode + ["Severity"]
    sub = df[[c for c in all_cols if c in df.columns]].dropna()

    # Encode categoricals
    le_map = {}
    for c in cat_cols_to_encode:
        if c in sub.columns:
            le = LabelEncoder()
            sub[c] = le.fit_transform(sub[c].astype(str))
            le_map[c] = le

    X_cols = [c for c in feature_cols + bool_cols + cat_cols_to_encode if c in sub.columns]
    X = sub[X_cols].astype(float)
    y = sub["Severity"] - 1  # make 0-indexed for XGBoost

    return X, y, le_map


def train_xgboost(X, y, test_size=0.2):
    """Train XGBoost classifier and return model + metrics."""
    print("\n=== Training XGBoost Severity Classifier ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=int(y.nunique()),
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=50)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("XGBoost Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("  ✓ confusion_matrix.png saved")

    # Feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, ax=ax, max_num_features=15, importance_type="gain",
                        color="steelblue")
    ax.set_title("XGBoost Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("  ✓ feature_importance.png saved")

    return model, X_test, y_test, {"accuracy": acc, "f1": f1}


# ===================================================================
# 7. EXPLAINABLE AI — SHAP
# ===================================================================
def explain_with_shap(model, X_test, max_display=15):
    """Generate SHAP summary and bar plots."""
    print("\n=== SHAP Explainability ===")
    sample = X_test.iloc[:2000]
    explainer = shap.Explainer(model, sample)
    shap_values = explainer(sample)

    # Summary plot (beeswarm)
    fig = plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_values, sample, max_display=max_display,
                      show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "shap_summary.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  ✓ shap_summary.png saved")

    # Mean absolute SHAP bar plot
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, plot_type="bar",
                      max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "shap_bar.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  ✓ shap_bar.png saved")

    return shap_values


# ===================================================================
# 8. MAIN PIPELINE
# ===================================================================
def main():
    # For demonstration, load a manageable sample (500k rows).
    # Set nrows=None or remove the argument to process the full 7.7M dataset.
    df = load_data(nrows=500_000)
    df = clean_data(df)
    df = engineer_features(df)

    # EDA
    run_all_eda(df)

    # Geospatial
    create_heatmap(df)
    dbscan_hotspots(df)

    # ML
    X, y, le_map = prepare_ml_data(df)
    model, X_test, y_test, metrics = train_xgboost(X, y)

    # SHAP
    explain_with_shap(model, X_test)

    print("\n✅ Full analysis pipeline complete!")
    print(f"   All visuals saved to: {VISUALS_DIR}")


if __name__ == "__main__":
    main()
