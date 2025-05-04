from pydantic import BaseModel, field_validator
from typing import List

class GarmentRequest(BaseModel):
    gender: str
    garment: str
    betas: List[float]
    gammas: List[float]

    @field_validator("betas")
    def validate_betas(cls, v):
        for beta in v:
            if not (-2.5 <= beta <= 2.5):
                raise ValueError("Each beta value must be between -2.5 and 2.5.")
        return v

    @field_validator("gammas")
    def validate_gammas(cls, v):
        for gamma in v:
            if not (-1.0 <= gamma <= 1.0):
                raise ValueError("Each gamma value must be between -1 and 1.")
        return v


class BodyMeshRequest(BaseModel):
    gender: str
    betas: List[float]

    @field_validator("betas")
    def validate_betas(cls, v):
        for beta in v:
            if not (-2.5 <= beta <= 2.5):
                raise ValueError("Each beta value must be between -2.5 and 2.5.")
        return v
