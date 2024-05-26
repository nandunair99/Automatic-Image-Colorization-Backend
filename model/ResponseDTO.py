from pydantic import BaseModel

class ResponseDTO(BaseModel):
    message: str | None=None
    status_code:int
    content: str