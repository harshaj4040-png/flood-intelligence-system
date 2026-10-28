import json
import math
from typing import Dict, List, Any
from app.config import GRID_DATA_FILE, RISK_THRESHOLD_LOW, RISK_THRESHOLD_CRITICAL, RESOURCE_DEMAND_RATES

class RiskScoringEngine:
    def __init__(self, config_path=GRID_DATA_FILE):
        with open(config_path, "r") as f:
            self.raw_data = json.load(f)
        self.zones = {z["id"]: z for z in self.raw_data["zones"]}

    def calculate_zone_risks(
    self,
    sensor_telemetry: Dict[str, Any],
    ai_prediction_by_zone: Dict[str, float] = None
) -> Dict[str, Any]:
        """
        Calculates Risk Severity Score (0.0 to 1.0) and translates risk to resource demands.
        """
        sensors_dict = sensor_telemetry.get("sensors", {})
        
        # Map sensor readings to zone
        zone_sensor_map = {}
        for s_id, s_data in sensors_dict.items():
            z_id = s_data["zone_id"]
            if z_id not in zone_sensor_map:
                zone_sensor_map[z_id] = []
            zone_sensor_map[z_id].append(s_data)

        risk_results = {}

        for zone_id, zone_info in self.zones.items():
            zone_sensors = zone_sensor_map.get(zone_id, [])
            
            if zone_sensors:
                # Average or max metrics across sensors in this zone
                max_water_level = max(s["water_level_m"] for s in zone_sensors)
                flood_stage = zone_info["flood_stage_m"]
                water_ratio = min(2.0, max_water_level / max(0.1, flood_stage))
                
                max_rate_of_rise = max(s["rate_of_rise_m_hr"] for s in zone_sensors)
                rate_factor = min(1.0, max(0.0, max_rate_of_rise / 2.0))
                
                rainfall = max(s["rainfall_mm_hr"] for s in zone_sensors)
                rain_factor = min(1.0, rainfall / 80.0)
            else:
                water_ratio = 0.0
                rate_factor = 0.0
                rain_factor = 0.0
                max_water_level = 0.0

            # Population & Elevation Vulnerability Score
            pop = zone_info.get("population", 10000)
            elev = zone_info.get("elevation_m", 205.0)
            
            # Lower elevation + higher density = higher vulnerability
            vuln_score = min(1.0, (pop / 40000.0) * 0.5 + max(0.0, (215.0 - elev) / 15.0) * 0.5)

            # Composite Risk Score Calculation (0.0 to 1.0)
            composite_risk = (
                0.50 * min(1.0, water_ratio) +
                0.20 * rate_factor +
                0.15 * rain_factor +
                0.15 * vuln_score
            )
            composite_risk = round(min(1.0, max(0.0, composite_risk)), 3)

            # Combine sensor-based risk with AI prediction
            if ai_prediction_by_zone and zone_id in ai_prediction_by_zone:
                ai_risk = ai_prediction_by_zone[zone_id] / 100.0

                composite_risk = (
                    0.7 * composite_risk +
                    0.3 * ai_risk
                )

                composite_risk = round(
                    min(1.0, max(0.0, composite_risk)), 3
                )

            # Risk Category Assignment
            if composite_risk >= RISK_THRESHOLD_CRITICAL:
                severity = "Critical"
                color = "#FF2E93"
            elif composite_risk >= RISK_THRESHOLD_LOW:
                severity = "Medium"
                color = "#FFA500"
            else:
                severity = "Low"
                color = "#00E676"
            # Calculate Resource Demand for this zone
            # Scale demands with population and severity multiplier
            pop_k = pop / 1000.0  # Population in thousands
            
            demands = {}
            for r_type, rates in RESOURCE_DEMAND_RATES.items():
                if severity == "Low" and composite_risk < 0.2:
                    demands[r_type] = 0
                else:
                    raw_demand = rates["base"] + (rates["pop_factor"] * pop_k * composite_risk)
                    if severity == "Critical":
                        raw_demand += rates["min_critical"]
                    demands[r_type] = math.ceil(raw_demand)

            # Inundation Depth on roads in this zone
            road_flood_depth_m = max(0.0, round(max_water_level - (flood_stage * 0.8), 2))

            risk_results[zone_id] = {
                "zone_id": zone_id,
                "zone_name": zone_info["name"],
                "lat": zone_info["lat"],
                "lon": zone_info["lon"],
                "polygon": zone_info["polygon"],
                "population": pop,
                "elevation_m": elev,
                "water_level_m": max_water_level,
                "flood_stage_m": flood_stage,
                "road_flood_depth_m": road_flood_depth_m,
                "composite_risk_score": composite_risk,
                "severity": severity,
                "color_code": color,
                "demands": demands
            }

        return risk_results
