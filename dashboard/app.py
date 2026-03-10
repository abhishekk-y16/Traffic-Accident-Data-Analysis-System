"""
Traffic Accident Data Analysis — Interactive Streamlit Dashboard
Run:  streamlit run dashboard/app.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
DATA_DIR = os.path.join(BASE_DIR, "data")

LOCAL_CSV = os.path.join(DATA_DIR, "US_Accidents_March23.csv")
KAGGLE_CACHE = os.path.join(
    os.path.expanduser("~"),
    ".cache", "kagglehub", "datasets",
    "sobhanmoosavi", "us-accidents", "versions", "13",
    "US_Accidents_March23.csv",
)
DATASET_PATH = LOCAL_CSV if os.path.exists(LOCAL_CSV) else KAGGLE_CACHE

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Traffic Accident Analysis Dashboard",
    page_icon="🚦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading accident data …")
def load_data(nrows=500_000):
    df = pd.read_csv(DATASET_PATH, nrows=nrows)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["Hour"] = df["Start_Time"].dt.hour
    df["DayOfWeek"] = df["Start_Time"].dt.day_name()
    df["Month"] = df["Start_Time"].dt.month
    df["Year"] = df["Start_Time"].dt.year
    df["Is_Weekend"] = df["DayOfWeek"].isin(["Saturday", "Sunday"]).astype(int)
    df["Place"] = (
        df["Street"].fillna("") + ", " +
        df["City"].fillna("") + ", " +
        df["State"].fillna("")
    )
    return df


df = load_data()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 Filters")

severity_range = st.sidebar.slider(
    "Severity", int(df["Severity"].min()), int(df["Severity"].max()),
    (int(df["Severity"].min()), int(df["Severity"].max()))
)

years = sorted(df["Year"].dropna().unique())
selected_years = st.sidebar.multiselect("Year(s)", years, default=years)

states = sorted(df["State"].dropna().unique())
selected_states = st.sidebar.multiselect("State(s)", states, default=[])

# Apply filters
mask = (
    df["Severity"].between(*severity_range) &
    df["Year"].isin(selected_years)
)
if selected_states:
    mask &= df["State"].isin(selected_states)
fdf = df[mask]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🚦 Traffic Accident Data Analysis Dashboard")
st.markdown("Interactive exploration of US traffic accidents — "
            "temporal patterns, geographic hotspots, weather impact, and severity insights.")

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Accidents", f"{len(fdf):,}")
c2.metric("Avg Severity", f"{fdf['Severity'].mean():.2f}")
c3.metric("States Covered", f"{fdf['State'].nunique()}")
c4.metric("Cities Covered", f"{fdf['City'].nunique():,}")

st.divider()

# ---------------------------------------------------------------------------
# TAB layout
# ---------------------------------------------------------------------------
tab_temporal, tab_geo, tab_weather, tab_severity, tab_map = st.tabs(
    ["⏰ Temporal", "📍 Geographic", "🌦 Weather", "⚠️ Severity", "🗺 Map"]
)

# ---- Temporal -----------------------------------------------------------
with tab_temporal:
    col1, col2 = st.columns(2)
    with col1:
        hourly = fdf["Hour"].value_counts().sort_index().reset_index()
        hourly.columns = ["Hour", "Count"]
        fig = px.line(hourly, x="Hour", y="Count",
                      title="Accidents by Hour of Day",
                      markers=True, color_discrete_sequence=["#EF553B"])
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
        daily = fdf["DayOfWeek"].value_counts().reindex(day_order).reset_index()
        daily.columns = ["Day", "Count"]
        fig = px.bar(daily, x="Day", y="Count",
                     title="Accidents by Day of Week",
                     color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        monthly = fdf.groupby("Month").size().reset_index(name="Count")
        fig = px.bar(monthly, x="Month", y="Count",
                     title="Monthly Accident Count",
                     color_discrete_sequence=["#FF7F0E"])
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        yearly = fdf.groupby("Year").size().reset_index(name="Count")
        fig = px.bar(yearly, x="Year", y="Count",
                     title="Yearly Accident Trend",
                     color_discrete_sequence=["#00CC96"])
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap: Hour vs Day
    st.subheader("Heatmap: Hour × Day of Week")
    pivot = fdf.pivot_table(index="DayOfWeek", columns="Hour",
                            values="Severity", aggfunc="count")
    pivot = pivot.reindex(day_order)
    fig = px.imshow(pivot, aspect="auto",
                    color_continuous_scale="YlOrRd",
                    labels=dict(color="Accidents"))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ---- Geographic ---------------------------------------------------------
with tab_geo:
    col1, col2 = st.columns(2)
    with col1:
        top_states = fdf["State"].value_counts().head(15).reset_index()
        top_states.columns = ["State", "Count"]
        fig = px.bar(top_states, y="State", x="Count", orientation="h",
                     title="Top 15 States by Accident Count",
                     color="Count", color_continuous_scale="Reds")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_cities = fdf["City"].value_counts().head(15).reset_index()
        top_cities.columns = ["City", "Count"]
        fig = px.bar(top_cities, y="City", x="Count", orientation="h",
                     title="Top 15 Cities by Accident Count",
                     color="Count", color_continuous_scale="Blues")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    if "Sunrise_Sunset" in fdf.columns:
        st.subheader("Day vs Night Accidents")
        dn = fdf["Sunrise_Sunset"].value_counts().reset_index()
        dn.columns = ["Period", "Count"]
        fig = px.pie(dn, values="Count", names="Period",
                     color_discrete_sequence=["#FECB52", "#1F77B4"])
        st.plotly_chart(fig, use_container_width=True)

# ---- Weather ------------------------------------------------------------
with tab_weather:
    col1, col2 = st.columns(2)
    with col1:
        top_weather = fdf["Weather_Condition"].value_counts().head(15).reset_index()
        top_weather.columns = ["Condition", "Count"]
        fig = px.bar(top_weather, y="Condition", x="Count", orientation="h",
                     title="Top 15 Weather Conditions During Accidents",
                     color="Count", color_continuous_scale="Teal")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(fdf, x="Temperature(F)", nbins=60,
                           title="Temperature Distribution During Accidents",
                           color_discrete_sequence=["salmon"])
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(fdf, x="Visibility(mi)", nbins=40,
                           title="Visibility Distribution",
                           color_discrete_sequence=["mediumpurple"])
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.histogram(fdf, x="Humidity(%)", nbins=40,
                           title="Humidity Distribution",
                           color_discrete_sequence=["mediumseagreen"])
        st.plotly_chart(fig, use_container_width=True)

# ---- Severity -----------------------------------------------------------
with tab_severity:
    col1, col2 = st.columns(2)
    with col1:
        sev = fdf["Severity"].value_counts().sort_index().reset_index()
        sev.columns = ["Severity", "Count"]
        fig = px.bar(sev, x="Severity", y="Count",
                     title="Severity Distribution",
                     color="Severity",
                     color_continuous_scale="OrRd")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sev_hour = fdf.groupby(["Hour", "Severity"]).size().reset_index(name="Count")
        fig = px.line(sev_hour, x="Hour", y="Count", color="Severity",
                      title="Severity by Hour of Day",
                      color_discrete_sequence=px.colors.sequential.YlOrRd)
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Severity by Weather Condition (Top 10)")
    top10_wc = fdf["Weather_Condition"].value_counts().head(10).index
    sub = fdf[fdf["Weather_Condition"].isin(top10_wc)]
    sev_wx = sub.groupby(["Weather_Condition", "Severity"]).size().reset_index(name="Count")
    fig = px.bar(sev_wx, x="Weather_Condition", y="Count", color="Severity",
                 barmode="stack", color_continuous_scale="Sunset")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ---- Map ----------------------------------------------------------------
with tab_map:
    st.subheader("Accident Heatmap (sampled)")
    sample_size = st.slider("Sample size for map", 1000, 50000, 10000, step=1000)
    sample = fdf.dropna(subset=["Start_Lat", "Start_Lng"]).sample(
        n=min(sample_size, len(fdf)), random_state=42
    )
    m = folium.Map(location=[39.5, -98.35], zoom_start=4,
                   tiles="CartoDB dark_matter")
    HeatMap(
        sample[["Start_Lat", "Start_Lng"]].values.tolist(),
        radius=6, blur=10, max_zoom=13
    ).add_to(m)
    st_folium(m, width=1200, height=600)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("Data source: US-Accidents (Sobhan Moosavi) · Built with Streamlit, Plotly & Folium")
