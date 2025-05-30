from pydantic import BaseModel, Field
from typing import Optional, List, Tuple


class Triple(BaseModel):
    subject: Optional[str] = Field(default=None)
    predicate: Optional[str] = Field(default=None)
    object: Optional[str] = Field(default=None)

    def normalize(self):
        return Triple(subject=self.subject.strip().lower() if self.subject is not None else None,
                      predicate=self.predicate.strip().lower() if self.predicate is not None else None,
                      object=self.object.strip().lower() if self.object is not None else None)


class Output(BaseModel):
    triples: List[Triple]


class SubjectMetadata(BaseModel):
    subject: str
    index: int
    url: str
    chunk_number: int
    triple: Tuple[str, str, str]
