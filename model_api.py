"""
model_api.py — FlightLens REST API
Wraps infer.py and exposes a /predict endpoint for the frontend.

Run with:
    uvicorn model_api:app --host 0.0.0.0 --port 8000 --reload

Requirements:
    pip install fastapi uvicorn pandas numpy scikit-learn xgboost
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from infer import predict_route, predict_itineraries, MODEL   # all logic lives in infer.py

app = FastAPI(title="FlightLens Prediction API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your domain in production
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

VALID_ORIGINS = {
    "LAX","JFK","MIA","SFO","BOS","ATL","DFW",
    "DEN","EWR","PHL","DTW","LGA","IAD","OAK","CLT",
}

class RouteRequest(BaseModel):
    origin: str       # e.g. "LAX"
    destination: str  # always "ORD" for this app

@app.post("/predict")
def predict(req: RouteRequest):
    origin = req.origin.upper()
    dest   = req.destination.upper()

    if origin not in VALID_ORIGINS:
        raise HTTPException(status_code=400, detail=f"Unknown origin: {origin}")
    if dest != "ORD":
        raise HTTPException(status_code=400, detail="Only ORD supported as destination")

    return predict_route(origin, dest)

class ItineraryRequest(BaseModel):
    origin: str
    destination: str
    departure_date: str            # 'YYYY-MM-DD'
    departure_hour: float = 12.0
    is_nonstop: bool = True
    num_segments: int = 1
    duration_minutes: Optional[float] = None
    is_basic_economy: bool = False
    airline_name: str = "United"
    current_price: Optional[float] = None


class ItinerariesRequest(BaseModel):
    items: List[ItineraryRequest]


@app.post("/predict_itineraries")
def predict_itineraries_route(req: ItinerariesRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="items must be non-empty")
    if len(req.items) > 25:
        raise HTTPException(status_code=400, detail="too many items (max 25)")
    return {"results": predict_itineraries([it.model_dump() for it in req.items])}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.get("/")
def root():
    return {"message": "FlightLens Prediction API v2", "docs": "/docs"}
