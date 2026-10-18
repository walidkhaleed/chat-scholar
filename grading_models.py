
from typing import Literal
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    model_validator
)


class RubricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str = Field(
        min_length=1,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$"
    )
    name: str = Field(min_length=1, max_length=150)
    description: str = Field(min_length=10, max_length=1500)
    weight: float = Field(gt=0, le=100)
    max_score: float = Field(gt=0, le=100)


class RubricCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    instructions: str = Field(
        default="",
        max_length=3000
    )
    criteria: list[RubricCriterion] = Field(
        min_length=1,
        max_length=15
    )

    @model_validator(mode="after")
    def validate_rubric(self):
        ids = [c.criterion_id for c in self.criteria]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Criterion IDs must be unique."
            )

        total = sum(c.weight for c in self.criteria)

        if abs(total - 100.0) > 0.001:
            raise ValueError(
                "Rubric weights must total 100%."
            )

        return self


class EssayTextRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=18000)


class GradeRequest(BaseModel):
    essay_id: str
    rubric_id: str


class CriterionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    score: float
    justification: str = Field(min_length=10)
    evidence_quotes: list[str] = Field(
        default_factory=list,
        max_length=3
    )
    improvement: str = Field(min_length=10)


class LLMGradeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criteria: list[CriterionAssessment]
    overall_feedback: str = Field(min_length=10)
    strengths: list[str]
    improvements: list[str]


class GradeStatus(BaseModel):
    status: Literal["completed"] = "completed"
