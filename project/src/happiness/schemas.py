from pydantic import BaseModel, Field

class HappinessInput(BaseModel):
    Log_GDP_per_capita: float = Field(..., ge=0, le=15)
    Social_support: float = Field(..., ge=0, le=1)
    Healthy_life_expectancy_at_birth: float = Field(..., ge=0, le=100)
    Freedom_to_make_life_choices: float = Field(..., ge=0, le=1)
    Generosity: float = Field(..., ge=-0.5, le=1)
    Perceptions_of_corruption: float = Field(..., ge=0, le=1)
    Positive_affect: float = Field(..., ge=0, le=1)
    Negative_affect: float = Field(..., ge=0, le=1)

class HappinessOutput(BaseModel):
    life_ladder: float