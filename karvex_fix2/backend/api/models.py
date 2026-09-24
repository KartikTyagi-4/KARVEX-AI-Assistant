from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    success: bool
    query: str
    answer: str
    error: str | None = None