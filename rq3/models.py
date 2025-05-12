from pydantic import BaseModel
from typing import List
from llama_index.core.node_parser import SentenceSplitter


class SentimentScore(BaseModel):
    url: str
    score: float


class SentimentScores(BaseModel):
    scores: List[SentimentScore]
    avg_score_disinformation: float
    avg_score_no_disinformation: float


class PromptTemplate(BaseModel):
    url: str
    article_text: str
    ground_truth_disinformation: str
    scores: List[float]

    def get_score(self, score: float) -> str:
        return f"{score:.2f}"

    def get_context(self) -> str:
        return f"""
3 ML models calculated a sentiment score for the article you must classify.
Each score is a number between -1 and 1.
-1 indicates a negative sentiment, 0 a neutral sentiment and 1 a positive sentiment.
Scores other than -1, 0 and 1 are also possible, but the range will always be between -1 and 1.
The scores given by the models for the article, separated by a comma, are : {",".join(map(lambda s: self.get_score(s), self.scores))}.
            """

    def get_article_text(self, max_words: int = 2500) -> str:
        splitter = SentenceSplitter(chunk_size=max_words, chunk_overlap=0)
        print(f"get article_text with max_words {max_words}", flush=True)
        return next(iter(splitter.split_text(self.article_text)))


class PromptTemplates(BaseModel):
    templates: List[PromptTemplate]


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
