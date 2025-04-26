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


class Metadata(BaseModel):
    ground_truth_url: str
    ground_truth_disinformation: str
    ground_truth_translated_text: str
    chunked_db_row_id: int
    chunk_text: str
    triple: int
    triple_text: str


class MetadataList(BaseModel):
    metadata: List[Metadata]


class SearchResult(BaseModel):
    url: str
    training_nearest_embedding_positions: List[int]
    training_nearest_embedding_scores: List[float]


class PromptTemplate(BaseModel):
    url: str
    article_text: str
    ground_truth_disinformation: str
    metadata: List[Metadata]
    scores: List[float]


class PromptTemplates(BaseModel):
    templates_article: List[PromptTemplate]
    templates_chunk: List[PromptTemplate]
    templates_triple: List[PromptTemplate]
