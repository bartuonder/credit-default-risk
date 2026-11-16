from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from credit_risk import schema

AMOUNT_MAX = 10_000_000
PayStatus = Field(
    ge=-2, le=9, description="-2 no consumption, -1 paid duly, 0 revolving, 1-9 months late"
)

EXAMPLE_LOW_RISK = {
    "limit_bal": 200000,
    "sex": 2,
    "education": 2,
    "marriage": 1,
    "age": 38,
    "pay_1": -1,
    "pay_2": -1,
    "pay_3": -1,
    "pay_4": -1,
    "pay_5": -1,
    "pay_6": -1,
    "bill_amt1": 8000,
    "bill_amt2": 7500,
    "bill_amt3": 9000,
    "bill_amt4": 8200,
    "bill_amt5": 7800,
    "bill_amt6": 8100,
    "pay_amt1": 8000,
    "pay_amt2": 7500,
    "pay_amt3": 9000,
    "pay_amt4": 8200,
    "pay_amt5": 7800,
    "pay_amt6": 8100,
}
EXAMPLE_HIGH_RISK = {
    "limit_bal": 30000,
    "sex": 1,
    "education": 3,
    "marriage": 2,
    "age": 27,
    "pay_1": 2,
    "pay_2": 2,
    "pay_3": 2,
    "pay_4": 1,
    "pay_5": 0,
    "pay_6": 0,
    "bill_amt1": 29500,
    "bill_amt2": 28900,
    "bill_amt3": 28000,
    "bill_amt4": 27000,
    "bill_amt5": 25500,
    "bill_amt6": 24000,
    "pay_amt1": 0,
    "pay_amt2": 1500,
    "pay_amt3": 0,
    "pay_amt4": 1200,
    "pay_amt5": 1000,
    "pay_amt6": 1000,
}


class Applicant(BaseModel):
    """One credit card client; field names follow the dataset columns in snake case."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_LOW_RISK]})

    limit_bal: int = Field(ge=1, le=AMOUNT_MAX, description="Credit limit in NT dollars")
    sex: Literal[1, 2] = Field(description="1 male, 2 female")
    education: Literal[1, 2, 3, 4] = Field(
        description="1 graduate school, 2 university, 3 high school, 4 other"
    )
    marriage: Literal[1, 2, 3] = Field(description="1 married, 2 single, 3 other")
    age: int = Field(ge=schema.AGE_RANGE[0], le=schema.AGE_RANGE[1], description="Age in years")
    pay_1: int = PayStatus
    pay_2: int = PayStatus
    pay_3: int = PayStatus
    pay_4: int = PayStatus
    pay_5: int = PayStatus
    pay_6: int = PayStatus
    bill_amt1: int = Field(ge=-AMOUNT_MAX, le=AMOUNT_MAX, description="Latest bill statement")
    bill_amt2: int = Field(ge=-AMOUNT_MAX, le=AMOUNT_MAX)
    bill_amt3: int = Field(ge=-AMOUNT_MAX, le=AMOUNT_MAX)
    bill_amt4: int = Field(ge=-AMOUNT_MAX, le=AMOUNT_MAX)
    bill_amt5: int = Field(ge=-AMOUNT_MAX, le=AMOUNT_MAX)
    bill_amt6: int = Field(ge=-AMOUNT_MAX, le=AMOUNT_MAX, description="Oldest bill statement")
    pay_amt1: int = Field(ge=0, le=AMOUNT_MAX, description="Latest payment")
    pay_amt2: int = Field(ge=0, le=AMOUNT_MAX)
    pay_amt3: int = Field(ge=0, le=AMOUNT_MAX)
    pay_amt4: int = Field(ge=0, le=AMOUNT_MAX)
    pay_amt5: int = Field(ge=0, le=AMOUNT_MAX)
    pay_amt6: int = Field(ge=0, le=AMOUNT_MAX, description="Oldest payment")


def to_frame(applicants: list[Applicant]) -> pd.DataFrame:
    """Model input frame with the dataset's column names."""
    rows = [a.model_dump() for a in applicants]
    frame = pd.DataFrame(rows).rename(columns=str.upper)
    return frame[schema.RAW_FEATURES].astype("int64")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicants: list[Applicant] = Field(min_length=1, description="Clients to score")


class Contribution(BaseModel):
    feature: str
    label: str
    value: str | int | float
    contribution: float = Field(description="Signed contribution to the model output")
    direction: Literal["increases risk", "decreases risk"]


class Prediction(BaseModel):
    default_probability: float = Field(ge=0, le=1)
    flagged: bool = Field(description="True when the probability reaches the decision threshold")
    decision: Literal["flag as likely default", "do not flag"]
    threshold: float
    risk_band: Literal["low", "medium", "high"]
    explanation: list[Contribution] | None = None


class PredictionResponse(Prediction):
    model_version: str
    explanation_unit: str | None = None


class BatchResponse(BaseModel):
    model_version: str
    count: int
    predictions: list[Prediction]
    explanation_unit: str | None = None


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    version: str


class ErrorDetail(BaseModel):
    field: str | None = None
    message: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] = []


class ErrorResponse(BaseModel):
    error: ErrorBody
