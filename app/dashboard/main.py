
import sys
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import json

from app.data_ingestion.sensor_feed import SensorDataIngestion
from app.data_ingestion.risk_scorer import RiskScoringEngine
from app.inventory.resource_manager import ResourceManager
from app.optimization.routing import GeospatialRoutingEngine
from app.optimization.solver import OptimizationSolver
from app.ml.predictor import predict_flood


# --------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Flood Intelligence System | Operations Command",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #E0E6ED;
    }

    .stMetric {
        background-color: #1E222D;
        padding: 14px;
        border-radius: 10px;
        border-left: 4px solid #00E676;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    .metric-critical {
        border-left-color: #FF2E93 !important;
    }

    .metric-warning {
        border-left-color: #FFA500 !important;
    }

    .stButton>button {
        background-color: #0066FF;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 16px;
        border: none;
        width: 100%;
    }

    .stButton>button:hover {
        background-color: #0052CC;
        box-shadow: 0 4px 12px rgba(0, 102, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# INITIALIZE SESSION STATE
# --------------------------------------------------
if "prediction_zone_id" not in st.session_state:
    st.session_state.prediction_zone_id = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "sensor_feed" not in st.session_state:
    st.session_state.sensor_feed = SensorDataIngestion()

if "risk_scorer" not in st.session_state:
    st.session_state.risk_scorer = RiskScoringEngine()

if "resource_mgr" not in st.session_state:
    st.session_state.resource_mgr = ResourceManager()

if "routing_engine" not in st.session_state:
    st.session_state.routing_engine = GeospatialRoutingEngine()

if "solver" not in st.session_state:
    st.session_state.solver = OptimizationSolver()

if "allocation_results" not in st.session_state:
    st.session_state.allocation_results = None


# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.title("⚡ Simulation Controls")
st.sidebar.markdown("---")

scenario = st.sidebar.selectbox(
    "🌧️ Storm Event Scenario",
    ["normal_storm", "mild_rain", "cloudburst", "dam_breach"],
    format_func=lambda x: {
        "mild_rain": "🌤️ Mild Rain Event",
        "normal_storm": "🌧️ Severe Monsoon Storm",
        "cloudburst": "⚡ Flash Flood / Cloudburst",
        "dam_breach": "🚨 Emergency Dam Breach"
    }[x]
)

sim_step = st.sidebar.slider(
    "⏱️ Simulation Time Step (T+Hours)",
    min_value=0,
    max_value=12,
    value=4,
    step=1
)

col_b1, col_b2 = st.sidebar.columns(2)

if col_b1.button("🎯 Optimize Allocation"):

    with st.spinner("Computing MILP Optimization & Routing..."):

        telemetry = st.session_state.sensor_feed.generate_telemetry(
            step=sim_step,
            scenario=scenario
        )

        ai_prediction_by_zone = {}

        if (
            st.session_state.prediction_result
            and st.session_state.prediction_zone_id
        ):
            ai_prediction_by_zone[
                st.session_state.prediction_zone_id
            ] = st.session_state.prediction_result["flood_probability"]

        zone_risks = st.session_state.risk_scorer.calculate_zone_risks(
            telemetry,
            ai_prediction_by_zone
        )

        transit_matrix = (
            st.session_state.routing_engine
            .compute_all_depot_zone_matrix(zone_risks)
        )

        depots = st.session_state.resource_mgr.get_all_depots()

        result = st.session_state.solver.solve_allocation(
            depots,
            zone_risks,
            transit_matrix
        )

        executed = st.session_state.resource_mgr.dispatch_resources(
            result["allocations"]
        )

        st.session_state.allocation_results = {
            "dispatches": result["allocations"],
            "status": result["solver_status"],
            "obj_val": result["objective_value"],
            "unserved": result["unserved_demand"]
        }

        st.sidebar.success(
            f"Dispatched {len(result['allocations'])} resource units!"
        )


if col_b2.button("🔄 Reset Depot Stock"):

    st.session_state.resource_mgr.reset_inventory()
    st.session_state.allocation_results = None

    st.sidebar.info("Depot stock reset to baseline.")


# --------------------------------------------------
# AI-BASED FLOOD PREDICTION
# --------------------------------------------------

st.markdown("---")
st.header("🌊 AI-Based Flood Prediction")

with st.form("flood_prediction_form"):

    st.subheader("Enter Environmental and Geographic Details")

    zone_options = {
        f"{zone['name']} ({zone_id})": zone_id
        for zone_id, zone in st.session_state.risk_scorer.zones.items()
    }

    selected_zone = st.selectbox(
        "Select Kerala Zone for Prediction",
        options=list(zone_options.keys())
    )

    selected_zone_id = zone_options[selected_zone]

    col1, col2, col3 = st.columns(3)

    with col1:

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=120.0
        )

        river_level = st.number_input(
            "River Level (m)",
            min_value=0.0,
            value=4.5
        )

        river_discharge = st.number_input(
            "River Discharge (m³/s)",
            min_value=0.0,
            value=250.0
        )

        soil_moisture = st.number_input(
            "Soil Moisture (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0
        )

    with col2:

        temperature = st.number_input(
            "Temperature (°C)",
            value=28.0
        )

        elevation = st.number_input(
            "Elevation (m)",
            min_value=0.0,
            value=900.0
        )

        slope = st.number_input(
            "Slope (degrees)",
            min_value=0.0,
            value=5.0
        )

        drainage = st.number_input(
            "Drainage Capacity (%)",
            min_value=0.0,
            max_value=100.0,
            value=30.0
        )

        urbanization = st.number_input(
            "Urbanization (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

    with col3:

        deforestation = st.number_input(
            "Deforestation (%)",
            min_value=0.0,
            max_value=100.0,
            value=40.0
        )

        population = st.number_input(
            "Population Density",
            min_value=0.0,
            value=5000.0
        )

        historical_floods = st.number_input(
            "Historical Floods",
            min_value=0.0,
            value=3.0
        )

        soil_type = st.selectbox(
            "Soil Type",
            ["Clay", "Sandy", "Loamy", "Silt"]
        )

        land_cover = st.selectbox(
            "Land Cover",
            ["Urban", "Forest", "Agricultural", "Water"]
        )

    predict_button = st.form_submit_button(
        "🔍 Predict Flood Risk"
    )


# Run prediction only when the form is submitted

if predict_button:

    input_data = {
        "Rainfall_mm": rainfall,
        "River_Level_m": river_level,
        "River_Discharge_m3s": river_discharge,
        "Soil_Moisture_pct": soil_moisture,
        "Humidity_pct": humidity,
        "Temperature_C": temperature,
        "Elevation_m": elevation,
        "Slope_deg": slope,
        "Drainage_Capacity_pct": drainage,
        "Urbanization_pct": urbanization,
        "Deforestation_pct": deforestation,
        "Population_Density": population,
        "Historical_Floods": historical_floods,
        "Soil_Type": soil_type,
        "Land_Cover": land_cover
    }

    try:

        with st.spinner("CatBoost model is predicting flood risk..."):

            st.session_state.prediction_result = predict_flood(
    input_data
)

        st.session_state.prediction_zone_id = selected_zone_id

    except Exception as e:

        st.session_state.prediction_result = None
        st.error(f"Prediction failed: {e}")


# Display saved prediction result outside the form

if st.session_state.prediction_result is not None:

    prediction_result = st.session_state.prediction_result

    st.subheader("AI Flood Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    result_col1.metric(
        "Flood Probability",
        f"{prediction_result['flood_probability']}%"
    )

    result_col2.metric(
        "Severity",
        prediction_result["severity"]
    )

    result_col3.metric(
        "Prediction",
        "Flood" if prediction_result["prediction"] == 1
        else "No Flood"
    )

    st.caption(
        f"Prediction model: {prediction_result['model']}"
    )


# --------------------------------------------------
# MAIN APPLICATION DATA FETCH
# --------------------------------------------------

telemetry = st.session_state.sensor_feed.generate_telemetry(
    step=sim_step,
    scenario=scenario
)

ai_prediction_by_zone = {}

if (
    st.session_state.prediction_result
    and st.session_state.prediction_zone_id
):
    ai_prediction_by_zone[
        st.session_state.prediction_zone_id
    ] = st.session_state.prediction_result["flood_probability"]

zone_risks = st.session_state.risk_scorer.calculate_zone_risks(
    telemetry,
    ai_prediction_by_zone
)

depots = st.session_state.resource_mgr.get_all_depots()


# --------------------------------------------------
# TITLE HEADER
# --------------------------------------------------

st.title(
    "🌊 Dynamic Flood Intelligence & Emergency Allocation System"
)

st.caption(
    f"Real-Time GIS Telemetry & Mathematical Optimization Engine "
    f"• Metro River Basin Command • T+{sim_step * 15} mins"
)


# --------------------------------------------------
# TOP KPI SUMMARY CARDS
# --------------------------------------------------

critical_count = sum(
    1 for z in zone_risks.values()
    if z["severity"] == "Critical"
)

medium_count = sum(
    1 for z in zone_risks.values()
    if z["severity"] == "Medium"
)

pop_at_risk = sum(
    z["population"]
    for z in zone_risks.values()
    if z["severity"] in ["Critical", "Medium"]
)

dispatched_units = (
    len(st.session_state.allocation_results["dispatches"])
    if st.session_state.allocation_results
    else 0
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric(
    "Critical Risk Zones",
    f"{critical_count} / {len(zone_risks)}",
    delta="Urgent Action" if critical_count > 0 else "Normal",
    delta_color="inverse"
)

kpi2.metric(
    "Medium Risk Zones",
    f"{medium_count}",
    delta_color="normal"
)

kpi3.metric(
    "Population at Risk",
    f"{pop_at_risk:,}",
    delta="Inundated" if pop_at_risk > 0 else "Safe"
)

kpi4.metric(
    "Active Dispatches",
    f"{dispatched_units} Units",
    delta="Optimized" if dispatched_units > 0 else "Idle"
)

kpi5.metric(
    "Solver Status",
    st.session_state.allocation_results["status"]
    if st.session_state.allocation_results
    else "Ready",
    delta="MILP CBC"
)

st.markdown("---")


# --------------------------------------------------
# GIS MAP AND RISK METRICS
# --------------------------------------------------

map_col, metrics_col = st.columns([7, 5])

with map_col:

    st.subheader(
        "🗺️ Live Geospatial Flood Risk & Dispatch Map"
    )

    m = folium.Map(
    location=[28.6139, 77.2090],
    zoom_start=12,
    tiles="OpenStreetMap"
)
    

    # Risk Zone Polygons

    for z_id, z in zone_risks.items():

        folium.Polygon(
            locations=z["polygon"],
            color=z["color_code"],
            weight=3,
            fill=True,
            fill_color=z["color_code"],
            fill_opacity=0.4,
            popup=folium.Popup(f"""
                <div style="font-family: Arial; width: 200px;">
                    <h4>{z['zone_name']} ({z_id})</h4>
                    <b>Severity:</b>
                    <span style="color:{z['color_code']}">
                        {z['severity']}
                    </span><br>
                    <b>Risk Score:</b> {z['composite_risk_score']}<br>
                    <b>Water Level:</b>
                    {z['water_level_m']}m / {z['flood_stage_m']}m<br>
                    <b>Population:</b> {z['population']:,}<br>
                    <b>Road Flood Depth:</b>
                    {z['road_flood_depth_m']}m<br>
                    <hr>
                    <b>Demands:</b><br>
                    • Rescue Boats:
                    {z['demands'].get('rescue_boats', 0)}<br>
                    • EMT Teams:
                    {z['demands'].get('medical_teams', 0)}<br>
                    • Relief Kits:
                    {z['demands'].get('relief_kits', 0)}
                </div>
            """, max_width=250)
        ).add_to(m)

    # Resource Depot Markers

    for d_id, d in depots.items():

        inv_str = "<br>".join([
            f"• {k.replace('_', ' ').title()}: {v}"
            for k, v in d["current_inventory"].items()
        ])

        folium.Marker(
            location=[d["lat"], d["lon"]],
            popup=folium.Popup(
                f"<b>{d['name']} ({d_id})</b>"
                f"<br><br><b>Stock Available:</b><br>{inv_str}",
                max_width=250
            ),
            tooltip=f"Depot: {d['name']}",
            icon=folium.Icon(
                color="blue",
                icon="home",
                prefix="fa"
            )
        ).add_to(m)

    # Sensor Gauge Markers

    for s_id, s in telemetry["sensors"].items():

        sensor_color = (
            "red" if s["status"] == "CRITICAL"
            else "orange" if s["status"] == "WARNING"
            else "green"
        )

        folium.CircleMarker(
            location=[
                s["lat"] if "lat" in s else 28.6139,
                s["lon"] if "lon" in s else 77.2090
            ],
            radius=6,
            color=sensor_color,
            fill=True,
            fill_color=sensor_color,
            fill_opacity=0.9,
            popup=(
                f"<b>Sensor {s_id}: {s['sensor_name']}</b>"
                f"<br>Level: {s['water_level_m']}m "
                f"(Rate: {s['rate_of_rise_m_hr']}m/h)"
            )
        ).add_to(m)

    # Dispatch Routes

    if st.session_state.allocation_results:

        for alloc in st.session_state.allocation_results["dispatches"]:

            poly = alloc.get("polyline", [])

            if poly and len(poly) > 1:

                r_type = alloc["resource_type"]

                line_color = (
                    "#00E676"
                    if r_type == "rescue_boats"
                    else "#00B0FF"
                )

                folium.PolyLine(
                    locations=poly,
                    color=line_color,
                    weight=4,
                    opacity=0.85,
                    dash_array=(
                        "5, 10"
                        if r_type == "rescue_boats"
                        else None
                    ),
                    tooltip=(
                        f"Dispatch: {alloc['quantity']}x "
                        f"{r_type.replace('_', ' ')} "
                        f"to {alloc['zone_name']} "
                        f"(ETA: {alloc['eta_minutes']} min)"
                    )
                ).add_to(m)

    st_folium(m, width="100%", height=520)


# --------------------------------------------------
# ZONE RISK AND RESOURCE DEMAND
# --------------------------------------------------

with metrics_col:

    st.subheader(
        "📊 Zone Risk Severity & Resource Demand"
    )

    zone_table = []

    for z_id, z in zone_risks.items():

        zone_table.append({
            "Zone": z["zone_name"],
            "Severity": z["severity"],
            "Risk Score": z["composite_risk_score"],
            "Water Level": f"{z['water_level_m']} m",
            "Boats Req.": z["demands"].get("rescue_boats", 0),
            "EMT Req.": z["demands"].get("medical_teams", 0),
            "Relief Kits": z["demands"].get("relief_kits", 0)
        })

    df_zones = pd.DataFrame(zone_table)

    st.dataframe(
        df_zones,
        column_config={
            "Severity": st.column_config.SelectboxColumn(
                "Severity",
                options=["Low", "Medium", "Critical"]
            ),
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score",
                format="%.2f",
                min_value=0.0,
                max_value=1.0
            )
        },
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("🚛 Recommended Resource Dispatches")

    if (
        st.session_state.allocation_results
        and st.session_state.allocation_results["dispatches"]
    ):

        df_dispatch = pd.DataFrame(
            st.session_state.allocation_results["dispatches"]
        )

        df_disp_clean = df_dispatch[
            [
                "depot_name",
                "zone_name",
                "resource_type",
                "quantity",
                "eta_minutes"
            ]
        ].copy()

        df_disp_clean.columns = [
            "From Depot",
            "Target Zone",
            "Resource Type",
            "Qty",
            "ETA (min)"
        ]

        st.dataframe(
            df_disp_clean,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Click '🎯 Optimize Allocation' in the sidebar "
            "to solve MILP resource dispatch."
        )


# --------------------------------------------------
# HYDROGRAPH AND TELEMETRY ANALYSIS
# --------------------------------------------------

st.markdown("---")

st.subheader("📈 Sensor Hydrographs & Telemetry Analysis")

h_col1, h_col2 = st.columns(2)

with h_col1:

    st.markdown(
        "##### 🌊 Simulated River Gauge Water Level Hydrograph (T+0 to T+12)"
    )

    steps = list(range(0, 13))
    hydro_data = {}

    for step_i in steps:

        t_data = st.session_state.sensor_feed.generate_telemetry(
            step=step_i,
            scenario=scenario
        )

        for s_id, s in t_data["sensors"].items():

            if s_id not in hydro_data:
                hydro_data[s_id] = []

            hydro_data[s_id].append(s["water_level_m"])

    df_hydro = pd.DataFrame(
        hydro_data,
        index=[f"T+{i*15}m" for i in steps]
    )

    st.line_chart(df_hydro)


with h_col2:

    st.markdown(
        "##### 🏢 Depot Inventory Capacities & Utilization"
    )

    depot_chart_data = []

    for d_id, d in depots.items():

        for r_type, curr in d["current_inventory"].items():

            initial = d["initial_inventory"][r_type]
            dispatched = d["dispatched"][r_type]

            depot_chart_data.append({
                "Depot": d["name"].split()[0],
                "Resource": r_type.replace("_", " ").title(),
                "Available": curr,
                "Dispatched": dispatched
            })

    df_depot_chart = pd.DataFrame(depot_chart_data)

    st.bar_chart(
        df_depot_chart,
        x="Depot",
        y=["Available", "Dispatched"]
    )
