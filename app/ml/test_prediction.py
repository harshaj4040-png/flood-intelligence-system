
from app.ml.predictor import predict_flood

sample_data = {
    "Rainfall_mm": 120,
    "River_Level_m": 4.5,
    "River_Discharge_m3s": 250,
    "Soil_Moisture_pct": 80,
    "Humidity_pct": 85,
    "Temperature_C": 28,
    "Elevation_m": 900,
    "Slope_deg": 5,
    "Drainage_Capacity_pct": 30,
    "Urbanization_pct": 75,
    "Deforestation_pct": 40,
    "Population_Density": 5000,
    "Historical_Floods": 3,
    "Soil_Type": "Clay",
    "Land_Cover": "Urban"
}

result = predict_flood(sample_data)

print("Flood Prediction Result")
print("-----------------------")
print(result)
