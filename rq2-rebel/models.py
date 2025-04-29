from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Union
from unidecode import unidecode
from llama_index.core.node_parser import SentenceSplitter


Groupings = Union["article", "chunk", "triple"]


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

    def get_context(self, grouping: Groupings, max_words: int = 2500):
        if len(self.metadata) == 0:
            return "No additional context is available"

        size_per_metadata = int(max_words / len(self.metadata))
        print(f"get context for {grouping} with {len(self.metadata)} metadata, max_words {max_words} and size_per_metadata {size_per_metadata}", flush=True)
        splitter = SentenceSplitter(chunk_size=size_per_metadata, chunk_overlap=0)
        if grouping == "article":
            return "\n".join([f"Related article text:\n{next(iter(splitter.split_text(m.ground_truth_translated_text)))}" for m in self.metadata])
        elif grouping == "chunk":
            return "\n".join([f"Related paragraph text:\n{next(iter(splitter.split_text(m.chunk_text)))}" for m in self.metadata])
        elif grouping == "triple":
            return "\n".join([f"Related RDF triple text:\n{next(iter(splitter.split_text(m.triple_text)))}" for m in self.metadata])
        else:
            raise ValueError(f"Unknown grouping: {grouping}")

    def get_article_text(self, max_words: int = 2500) -> str:
        splitter = SentenceSplitter(chunk_size=max_words, chunk_overlap=0)
        print(f"get article_text with max_words {max_words}", flush=True)
        return next(iter(splitter.split_text(self.article_text)))


class PromptTemplates(BaseModel):
    templates_article: List[PromptTemplate]
    templates_chunk: List[PromptTemplate]
    templates_triple: List[PromptTemplate]

    def get_templates(self, grouping: Groupings) -> List[PromptTemplate]:
        if grouping == "article":
            return self.templates_article
        elif grouping == "chunk":
            return self.templates_chunk
        elif grouping == "triple":
            return self.templates_triple
        else:
            raise ValueError(f"Unknown grouping: {grouping}")


class ClassificationOutput(BaseModel):
    contains_disinformation: bool


class ModelResult(BaseModel):
    url: str
    result: str
    y: int


class ModelResults(BaseModel):
    results: List[ModelResult]


class ModelStats(BaseModel):
    model_name: str
    accuracy: str
    precision: str
    recall: str
    f1: str
