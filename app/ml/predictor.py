
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "catboost_flood_model.pkl")

model = joblib.load(MODEL_PATH)


def predict_flood(data):
    values = {
        "Rainfall_mm": float(data["Rainfall_mm"]),
        "River_Level_m": float(data["River_Level_m"]),
        "River_Discharge_m3s": float(data["River_Discharge_m3s"]),
        "Soil_Moisture_pct": float(data["Soil_Moisture_pct"]),
        "Humidity_pct": float(data["Humidity_pct"]),
        "Temperature_C": float(data["Temperature_C"]),
        "Elevation_m": float(data["Elevation_m"]),
        "Slope_deg": float(data["Slope_deg"]),
        "Drainage_Capacity_pct": float(data["Drainage_Capacity_pct"]),
        "Urbanization_pct": float(data["Urbanization_pct"]),
        "Deforestation_pct": float(data["Deforestation_pct"]),
        "Population_Density": float(data["Population_Density"]),
        "Historical_Floods": float(data["Historical_Floods"]),
        "Soil_Type": str(data["Soil_Type"]),
        "Land_Cover": str(data["Land_Cover"])
    }

    values["Drainage_Risk_pct"] = 100 - values["Drainage_Capacity_pct"]
    values["Flood_Risk_Index"] = (
        values["Rainfall_mm"] * values["River_Level_m"]
        * values["Soil_Moisture_pct"]
    )
    values["Drainage_Stress"] = (
        values["Urbanization_pct"] /
        (values["Drainage_Capacity_pct"] + 1)
    )
    values["Water_Load"] = (
        values["Rainfall_mm"] + values["River_Discharge_m3s"]
    )
    values["Terrain_Risk"] = (
        values["Slope_deg"] / (values["Elevation_m"] + 1)
    )
    values["Rainfall_River_Interaction"] = (
        values["Rainfall_mm"] * values["River_Level_m"]
    )
    values["Urban_Flood_Pressure"] = (
        values["Urbanization_pct"] * values["Population_Density"]
    )
    values["Environmental_Risk"] = (
        values["Deforestation_pct"] * values["Soil_Moisture_pct"]
    )
    values["Combined_Flood_Score"] = (
        values["Flood_Risk_Index"] + values["Water_Load"]
        + values["Drainage_Stress"]
    )

    df = pd.DataFrame([values], columns=model.feature_names_)

    prediction = int(model.predict(df)[0])
    probabilities = model.predict_proba(df)[0]
    flood_probability = round(float(probabilities[1]) * 100, 2)

    if flood_probability < 25:
        severity = "Low"
    elif flood_probability < 50:
        severity = "Moderate"
    elif flood_probability < 75:
        severity = "High"
    else:
        severity = "Very High"

    return {
        "prediction": prediction,
        "flood_probability": flood_probability,
        "severity": severity,
        "model": "CatBoost"
    }
