from pydantic import BaseModel, Field
from typing import Optional, List


class Triple(BaseModel):
    subject: Optional[str] = Field(default=None)
    predicate: Optional[str] = Field(default=None)
    object: Optional[str] = Field(default=None)


class Output(BaseModel):
    triples: List[Triple]
