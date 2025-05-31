from pydantic import BaseModel


class Answer(BaseModel):
    """
    Represents an answer in the Hato model.
    """

    observation: str
    reasoning: str
    ward: str
    town: str | None = None
    confidence: float | None = None
