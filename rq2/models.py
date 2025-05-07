from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import re


class Triple(BaseModel):
    subject: Optional[str] = Field(default=None)
    predicate: Optional[str] = Field(default=None)
    object: Optional[str] = Field(default=None)

    def is_valid(self):
        return (
            self.subject is not None
            and self.predicate is not None
            and self.object is not None
        )

    def __str__(self):
        if self.subject is None or self.object is None:
            return ""
        if self.predicate is None:
            return f"{self.subject} ~ is ~ {self.object}"
        return f"{self.subject} ~ {self.predicate} ~ {self.object}"

    def __repr__(self):
        return str(self)

    def normalize(self):
        return Triple(
            subject=self.subject.strip().lower() if self.subject is not None else None,
            predicate=self.predicate.strip().lower()
            if self.predicate is not None
            else None,
            object=self.object.strip().lower() if self.object is not None else None,
        )


def triple_comparator(triple: Triple) -> tuple:
    return (triple.subject, triple.predicate, triple.object)


class Output(BaseModel):
    triples: List[Triple]


class ModelInput(BaseModel):
    model_name: str
    model_params: Dict[str, Any]


class RowResult(BaseModel):
    url: str
    valid: bool
    result_ttl: str
    result_json: str
    y: int


class ModelResult(BaseModel):
    model_input: ModelInput
    row_results: List[RowResult]


def sanitize_filename(filename: str) -> str:
    return re.sub(r'[\/:*?"<>|]', "_", filename)


def remove_markdown(text: str) -> str:
    return text.replace("```turtle", "").replace("```json", "").replace("```", "")
