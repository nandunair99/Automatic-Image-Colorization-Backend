from pydantic import BaseModel


class ResponseDTO(BaseModel):
    message: str
