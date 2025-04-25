from pydantic import BaseModel, Field
from typing import Optional, List, Tuple
from unidecode import unidecode


class Triple(BaseModel):
    subject: Optional[str] = Field(default=None)
    predicate: Optional[str] = Field(default=None)
    object: Optional[str] = Field(default=None)

    def normalize(self):
        return Triple(
            subject=unidecode(self.subject.strip().lower()) if self.subject is not None else None,
            predicate=unidecode(self.predicate.strip().lower()) if self.predicate is not None else None,
            object=unidecode(self.object.strip().lower()) if self.object is not None else None,
        )


class TripleMetRowId(BaseModel):
    triple: Triple
    row_id: int


class Output(BaseModel):
    triples: List[Triple]


class SubjectMetadata(BaseModel):
    subject: str
    index: int
    url: str
    chunk_number: int
    triple: Tuple[str, str, str]


class Sample(BaseModel):
    # row_ids van de training set in de ground_truth file
    training: List[int]
    test: List[int]

    def __str__(self) -> str:
        return f"Sample, {len(self.training)} training, {len(self.test)} test"
