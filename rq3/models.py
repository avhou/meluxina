from pydantic import BaseModel
from typing import List


class SentimentScore(BaseModel):
    url: str
    score: float


class SentimentScores(BaseModel):
    scores: List[SentimentScore]
    avg_score_disinformation: float
    avg_score_no_disinformation: float
