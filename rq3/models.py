from pydantic import BaseModel
from typing import List


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


class PromptTemplates(BaseModel):
    templates: List[PromptTemplate]
