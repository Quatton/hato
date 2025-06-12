import json

from model.result import Results
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class Answer(BaseModel):
    ward: str
    town: str | None = None


class DatasetEntry(BaseModel):
    id: int
    panoId: str
    answer: Answer
    messages: list[ChatMessage]


with open("out/output_gpt.json", "r") as f:
    output_data = json.load(f)
    result = Results.model_validate(output_data)
    results = result.results

dataset_entries: list[DatasetEntry] = []
for result in results:
    id = result.index
    panoId = result.panoid
    answer = Answer(ward=result.answer.ward, town=result.answer.town)
    messages = [
        ChatMessage(role="user", content="<image>"),
        ChatMessage(role="assistant", content=result.answer.raw),
    ]
    dataset_entry = DatasetEntry(id=id, panoId=panoId, answer=answer, messages=messages)

    dataset_entries.append(dataset_entry)

with open("out/tokyo-2000-messages.json", "w") as f:
    json.dump(
        [entry.model_dump() for entry in dataset_entries],
        f,
        indent=2,
        ensure_ascii=False,
    )
    print(f"Dataset saved with {len(dataset_entries)} entries.")
