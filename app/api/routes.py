import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

from app.data_ingestion.sensor_feed import SensorDataIngestion
from app.data_ingestion.risk_scorer import RiskScoringEngine
from app.inventory.resource_manager import ResourceManager
from app.optimization.routing import GeospatialRoutingEngine
from app.optimization.solver import OptimizationSolver
from app.ml.predictor import predict_flood

class FloodPredictionRequest(BaseModel):
    Rainfall_mm: float
    River_Level_m: float
    River_Discharge_m3s: float
    Soil_Moisture_pct: float
    Humidity_pct: float
    Temperature_C: float
    Elevation_m: float
    Slope_deg: float
    Drainage_Capacity_pct: float
    Urbanization_pct: float
    Deforestation_pct: float
    Population_Density: float
    Historical_Floods: float
    Soil_Type: str
    Land_Cover: str


app = FastAPI(
    title="Kerala Flood Intelligence System API",
    description="Real-time flood prediction, risk scoring, and dynamic resource allocation API",
    version="1.0.0"
)

# Enable CORS for Vite frontend (http://localhost:5173, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services
sensor_feed = SensorDataIngestion()
risk_scorer = RiskScoringEngine()
resource_mgr = ResourceManager()
routing_engine = GeospatialRoutingEngine()
solver = OptimizationSolver()
# Dynamic Population-at-Risk Calculation
def calculate_dynamic_population_risk(
    baseline_population,
    baseline_water_level,
    current_water_level,
    alpha,
    evacuated_population
):
    """
    Calculates dynamic population at risk based on
    water-level increase and evacuation progress.

    P_exposed(t) = P0 * (1 + alpha * Δh)
    P_risk(t) = P_exposed(t) - min(P_exposed(t), Evacuated(t))
    """

    delta_h = max(0.0, current_water_level - baseline_water_level)

    exposed_population = round(
        baseline_population * (1 + alpha * delta_h)
    )

    evacuated_population = min(
        exposed_population,
        max(0, round(evacuated_population))
    )

    population_at_risk = (
        exposed_population - evacuated_population
    )

    return {
        "exposed_population": exposed_population,
        "evacuated_population": evacuated_population,
        "population_at_risk": population_at_risk,
        "delta_water_level": round(delta_h, 2)
    }

@app.get("/")
def read_root():
    return {
        "system": "Kerala Flood Intelligence System (FIS)",
        "status": "ONLINE",
        "modules": ["sensor_ingestion", "risk_scorer", "inventory_tracker", "milp_solver", "networkx_routing"]
    }

@app.get("/api/simulation")
@app.get("/api/v1/simulation")
def get_simulation_state(step: int = Query(2), scenario: str = Query("cloudburst")):
    """
    Returns full simulation payload for the requested timeline step (0: T+0, 1: T+15, 2: T+30, 3: T+45, 4: T+60)
    """
    # 1. Telemetry & Zone Risk Calculation
    telemetry = sensor_feed.generate_telemetry(
        step=step,
        scenario=scenario,
       )
    print("TELEMETRY DATA:", telemetry)

    zone_risks = risk_scorer.calculate_zone_risks(telemetry)

    # Generate T+0 telemetry to obtain baseline water levels
    baseline_telemetry = sensor_feed.generate_telemetry(
        step=0,
        scenario=scenario
    )

    baseline_zone_risks = risk_scorer.calculate_zone_risks(
        baseline_telemetry
    )
    
    # 2. Network Routing & Optimization
    transit_matrix = routing_engine.compute_all_depot_zone_matrix(zone_risks)
    depots = resource_mgr.get_all_depots()
    optimization_result = solver.solve_allocation(depots, zone_risks, transit_matrix)
    print("SOLVER ALLOCATIONS:", optimization_result["allocations"])
    dispatches = []
    # Group actual MILP allocations by resource type
    allocated_resources = {}

    for allocation in optimization_result["allocations"]:
        resource = allocation["resource_type"]
        quantity = allocation["quantity"]

        if resource not in allocated_resources:
            allocated_resources[resource] = 0

        allocated_resources[resource] += quantity
    # Build rich zone details matching reference layout
    zones_detail = {}
    total_people_at_risk = 0
    critical_count = 0
    
    for z_id, z in zone_risks.items():
        score = z["composite_risk_score"]
        if score < 0.3:
            z["severity"] = "Low"
        elif score < 0.6:
            z["severity"] = "Medium"
        else:
             z["severity"] = "Critical"
       

        if z["composite_risk_score"] >= 0.60:
            critical_count += 1

                # Dynamic population-at-risk calculation
        baseline_zone = baseline_zone_risks[z_id]

        baseline_population = z.get("population", 15000)
        baseline_water_level = baseline_zone["water_level_m"]
        current_water_level = z["water_level_m"]

        # Terrain exposure sensitivity
        # 0.25 for hilly/valley regions such as Wayanad
        # 0.35 for low-lying/coastal regions
        alpha = 0.25 if z_id == "Z3" else 0.35

        dynamic_population = calculate_dynamic_population_risk(
            baseline_population=baseline_population,
            baseline_water_level=baseline_water_level,
            current_water_level=current_water_level,
            alpha=alpha,
            evacuated_population=0
        )

        pop_risk = dynamic_population["population_at_risk"]
       # if z["severity"] == "Critical":
           # pop_risk = 18420
        #elif z["severity"] == "High":
          #  pop_risk = 22000

        total_people_at_risk += pop_risk

        zones_detail[z_id] = {
            "id": z_id,
            "name": z["zone_name"],
            "severity": z["severity"],
            "composite_risk_score": round(score, 2),
            "water_level_m": z["water_level_m"],
            "flood_stage_m": z["flood_stage_m"],
            "population_at_risk": pop_risk,
            "exposed_population": dynamic_population["exposed_population"],
            "evacuated_population": dynamic_population["evacuated_population"],
            "water_level_change_m": dynamic_population["delta_water_level"],
            "affected_villages": 12 if z_id == "Z3" else 8 if z_id == "Z2" else 4,
            "rainfall_24h_mm": 220 if z_id == "Z3" else 160 if z_id == "Z2" else 95,
            "status_text": "Heavy rainfall; possible landslides" if z_id == "Z3" else "Rising water levels" if z["severity"] in ["High", "Critical"] else "Monitoring river flow",
            "polygon": sensor_feed.zones[z_id]["polygon"],
            "lat": sensor_feed.zones[z_id]["lat"],
            "lon": sensor_feed.zones[z_id]["lon"],
            "demands": z["demands"]
        }

    # Static / Dynamic Alerts list
    alerts = []

    for z_id, z in zone_risks.items():
        if z["severity"] in ["High", "Critical"]:
            alerts.append({
            "id": z_id,
            "time": telemetry["timestamp"],
            "text": f'{z["zone_name"]} is at {z["severity"]} flood risk',
            "severity": z["severity"].upper(),
            "status": "ACTIVE",
        })
    # Transits & Dispatches for map vectors
        transit_routes = []
        for allocation in optimization_result["allocations"]:
            depot_id = allocation["depot_id"]
            zone_id = allocation["zone_id"]
            resource_type = allocation["resource_type"]

            route = transit_matrix.get(
            (depot_id, zone_id, resource_type)
        )

            if route and route.get("success", False):
                transit_routes.append({
                    "id": f"{depot_id}_{zone_id}_{resource_type}",
                    "from_depot": depot_id,
                    "from_name": allocation["depot_name"],
                    "from_coords": route.get("from_coords", []),
                    "to_zone": zone_id,
                    "to_name": allocation["zone_name"],
                    "to_coords": route.get("to_coords", []),
                    "polyline": route.get("polyline", []),
                    "asset": resource_type,
                    "resource_type": resource_type,
                    "eta_minutes": allocation["eta_minutes"],
                    "status": "En Route"
        })
        print("FINAL TRANSIT ROUTES:", transit_routes)
        print("TRANSIT ROUTES COUNT:", len(transit_routes))

    # Specific dispatches table for Z3 Wayanad
    z3_dispatches = [
        {"time": "16:30", "resource": "Rescue Boats", "quantity": 4, "from_depot": "D1 (Kochi)", "status": "En Route"},
        {"time": "15:52", "resource": "Medical Teams", "quantity": 4, "from_depot": "D1 (Kochi)", "status": "En Route"},
        {"time": "15:35", "resource": "Medicines", "quantity": 200, "from_depot": "D1 (Kochi)", "status": "Dispatched"},
        {"time": "15:27", "resource": "Water Pumps", "quantity": 2, "from_depot": "D1 (Kochi)", "status": "En Route"},
        {"time": "15:20", "resource": "Relief Shelters", "quantity": 1, "from_depot": "D1 (Kochi)", "status": "En Route"}
    ]

    # Allocated resource progress
    #allocated_resources = {
      #  "rescue_boats": {"allocated": 14, "total": 20, "percentage": 67, "status": "En Route"},
        #"medical_teams": {"allocated": 10, "total": 12, "percentage": 100, "status": "En Route"},
        ##"relief_shelters": {"allocated": 2, "total": 4, "percentage": 50, "status": "Preparing"}
    #}

    return {
        "step": step,
        "scenario": scenario,
        "timestamp": telemetry["timestamp"],
        "overall_stats": {
            "total_zones": 8,
           "people_at_risk": total_people_at_risk,
            #"people_at_risk_trend": "+12%",
            "critical_zones": critical_count,
           "active_dispatches": len(optimization_result["allocations"]),
           "active_alerts": sum(1 for alert in alerts if alert["status"] == "ACTIVE"),
            "weather": (
    f'{max(s["rainfall_mm_hr"] for s in telemetry["sensors"].values()):.1f} mm/hr '
    f'{"Heavy Rainfall" if max(s["rainfall_mm_hr"] for s in telemetry["sensors"].values()) >= 50 else "Moderate Rainfall" if max(s["rainfall_mm_hr"] for s in telemetry["sensors"].values()) >= 10 else "Light Rainfall"}'
),
        },
        "zones": zones_detail,
        "depots": depots,
        "alerts": alerts,
        "transit_routes": transit_routes,
        "allocated_resources": allocated_resources,
        "z3_dispatches": z3_dispatches,
        "total_allocated_sum": sum(allocated_resources.values()),
        "total_capacity_sum": sum(
            depot["initial_inventory"].get(resource, 0)
            for depot in depots.values()
            for resource in allocated_resources
),
"total_allocated_pct": round(
    (
        sum(allocated_resources.values())
        / sum(
            depot["initial_inventory"].get(resource, 0)
            for depot in depots.values()
            for resource in allocated_resources
        )
        * 100
    ),
    1
) if sum(
    depot["initial_inventory"].get(resource, 0)
    for depot in depots.values()
    for resource in allocated_resources
) > 0 else 0
}

@app.post("/api/predict")
def predict_flood_api(request: FloodPredictionRequest):
    try:
        result = predict_flood(request.model_dump())

        return {
            "status": "success",
            "result": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
