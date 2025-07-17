# app/schema.py

from pydantic import BaseModel

class PredictionInput(BaseModel):
    site_area: float
    structure_type: str
    water_consumption: float
    recycling_rate: float
    utilisation_rate: float
    air_quality_index: float
    issue_resolution_time: float
    resident_count: int
