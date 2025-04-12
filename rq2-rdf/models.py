from pydantic import BaseModel, Field
from typing import Optional, List, Tuple


class Triple(BaseModel):
    subject: Optional[str] = Field(default=None)
    predicate: Optional[str] = Field(default=None)
    object: Optional[str] = Field(default=None)


class Output(BaseModel):
    triples: List[Triple]


class SubjectMetadata(BaseModel):
    subject: str
    index: int
    url: str
    chunk_number: int
    triple: Tuple[str, str, str]
