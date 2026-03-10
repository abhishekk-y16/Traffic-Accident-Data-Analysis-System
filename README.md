# 🚦 Traffic Accident Data Analysis System

A comprehensive analytical framework for exploring, visualising, and predicting traffic accident severity using the **US-Accidents** dataset (~7.7 million records, Feb 2016 – Mar 2023).

## Features

| Module | Description |
|---|---|
| **Exploratory Data Analysis** | Temporal patterns (hourly, daily, seasonal), severity distribution, weather impact, top cities/states visualisations |
| **Geospatial Intelligence** | Interactive Folium heatmaps, DBSCAN‑based hotspot ("Black Spot") detection using haversine metric |
| **Machine Learning** | XGBoost severity classifier with baseline comparisons (Logistic Regression, Random Forest) |
| **Explainable AI** | SHAP beeswarm and bar plots for global/local feature importance |
| **Interactive Dashboard** | Streamlit app with filters, KPI cards, Plotly charts, and live Folium map |

## Project Structure

```
Traffic Accident Data Analysis System/
├── data/                          # Dataset directory
├── notebooks/
│   └── analysis.ipynb             # Full analysis Jupyter notebook
├── src/
│   └── analysis.py                # Reusable analysis pipeline (CLI)
├── dashboard/
│   └── app.py                     # Streamlit interactive dashboard
├── visuals/                       # Generated plots & maps
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn folium plotly streamlit scikit-learn xgboost shap kagglehub streamlit-folium
```

### 2. Download Dataset

```python
import kagglehub
path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
print("Path to dataset files:", path)
```

Or place `US_Accidents_March23.csv` directly in the `data/` folder.

### 3. Run the Analysis Script

```bash
python src/analysis.py
```

This executes the full pipeline: data loading → cleaning → EDA → geospatial mapping → XGBoost training → SHAP explainability. All plots are saved to `visuals/`.

### 4. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open the local URL shown in the terminal to access the interactive dashboard with filters, charts, and heatmap.

### 5. Explore the Notebook

Open `notebooks/analysis.ipynb` in Jupyter or VS Code and run cells sequentially.

## Key Insights

- **Peak hours:** Accidents concentrate during 14:00–17:00 (rush hour); severity peaks at midday and late night
- **Top locations:** California, Florida, and Texas lead; Miami, Houston, and Los Angeles dominate cities
- **Weather:** Rain, fog, and snow disproportionately increase severity despite fair weather having the most volume
- **XGBoost** outperforms Logistic Regression and Random Forest on tabular accident data
- **SHAP** reveals Hour, Distance, Visibility, and Temperature as the most influential predictors

## Dataset

**US-Accidents** by Sobhan Moosavi  
- ~7.7 million accident records across 49 US states  
- 46 features: severity, GPS coordinates, timestamps, weather, road infrastructure  
- Source: [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Plotly · Folium · Scikit-learn · XGBoost · SHAP · Streamlit
