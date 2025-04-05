from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Any, Optional, Callable
import re

PromptType = Literal["zero-shot", "one-shot", "few-shot"]

class Output(BaseModel):
    contains_disinformation: bool

class ModelInput(BaseModel):
    model_name: str
    model_params: Dict[str, Any]
    prompts: Dict[PromptType, str]
    prompt_generation: Callable[[str, str], str] = Field(default_factory=lambda: lambda prompt, text: "", exclude=True)
    model_creation: Callable[[Any], Any] = Field(default_factory=lambda: lambda input: None, exclude=True)

class RowResult(BaseModel):
    url: str
    invalid: bool
    result: str
    y: int
    y_hat: int

class ModelResult(BaseModel):
    model_input: ModelInput
    row_results: Dict[PromptType, List[RowResult]]

def sanitize_filename(filename: str) -> str:
    return re.sub(r'[\/:*?"<>|]', '_', filename)

class ModelStats(BaseModel):
    model_name: str
    accuracy: str
    precision: str
    recall: str
    f1: str


class TripleModelInput(BaseModel):
    model_name: str
    model_params: Dict[str, Any]

class TripleRowResult(BaseModel):
    url: str
    valid: bool
    result_ttl: str
    result_json: str
    y: int

class TripleModelResult(BaseModel):
    model_input: TripleModelInput
    row_results: List[TripleRowResult]

def remove_markdown(text: str) -> str:
    return (text
            .replace('```turtle', '')
            .replace('```json', '')
            .replace('```', ''))

class Triple(BaseModel):
    subject: Optional[str] = Field(default=None)
    predicate: Optional[str] = Field(default=None)
    object: Optional[str] = Field(default=None)

class TripleOutput(BaseModel):
    triples: List[Triple]
