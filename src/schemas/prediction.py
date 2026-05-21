from pydantic import BaseModel, Field
from typing import Dict

class ClassificationRequest(BaseModel):
    """Schema for prediction request."""
    text: str = Field(
        ...,
        max_length=500,
        description="The student concern or appointment reason."
    )

class ClassificationResponse(BaseModel):
    """Schema for prediction response."""
    level: str = Field(
        ...,
        description="The predicted urgency level (LOW, MEDIUM, HIGH, CRITICAL)."
    )
    confidence: float = Field(
        ...,
        description="The probability score associated with the level."
    )
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional prediction metadata."
    )
