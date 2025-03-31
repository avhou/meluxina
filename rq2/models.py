from pydantic import BaseModel
from typing import List, Dict, Any
import re


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str

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
    return re.sub(r'[\/:*?"<>|]', '_', filename)


def remove_markdown(text: str) -> str:
    return (text
            .replace('```turtle', '')
            .replace('```json', '')
            .replace('```', ''))
