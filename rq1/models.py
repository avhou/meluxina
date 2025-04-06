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
    prompt_generation: Callable[[str, str], str] = Field(
        default_factory=lambda: lambda prompt, text: "", exclude=True
    )
    model_creation: Callable[[Any], Any] = Field(
        default_factory=lambda: lambda input: None, exclude=True
    )


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
    return re.sub(r'[\/:*?"<>|]', "_", filename)


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
    return text.replace("```turtle", "").replace("```json", "").replace("```", "")


class Triple(BaseModel):
    subject: Optional[str] = Field(default=None)
    predicate: Optional[str] = Field(default=None)
    object: Optional[str] = Field(default=None)

    def __str__(self):
        if self.subject is None or self.object is None:
            return ""
        if self.predicate is None:
            return f"{self.subject} ~ is ~ {self.object}"
        return f"{self.subject} ~ {self.predicate} ~ {self.object}"

    def __repr__(self):
        return str(self)


class TripleOutput(BaseModel):
    triples: List[Triple]


def read_triple_map(triple_file: str) -> (str, Dict[str, List[Triple]]):
    print(f"reading triple file {triple_file}")
    with open(triple_file, "r") as f:
        triple_generation_model = os.path.splitext(os.path.basename(triple_file))[0]
        content = f.read()
        triple_model = TripleModelResult.model_validate_json(content)
        print(
            f"read {len(triple_model.row_results)} triples from {triple_file}, triple_generation_model is {triple_generation_model}",
            flush=True,
        )
        triple_map: Dict[str, List[Triple]] = {}
        for row in triple_model.row_results:
            try:
                triples = remove_markdown(row.result_json)
                triple_output = TripleOutput.model_validate_json(triples)
                triple_map[row.url] = triple_output.triples
            except Exception as e:
                print(f"could not parse url {row.url} with error {e}", flush=True)
        return (triple_generation_model, triple_map)
