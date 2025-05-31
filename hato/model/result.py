from typing import List
from pydantic import BaseModel

from hato.model.answer import Answer
from hato.model.dataset import Address


class ResultEntry(BaseModel):
    index: int
    panoid: str
    answer: Answer
    actual_address: Address


class Results(BaseModel):
    results: List[ResultEntry]
