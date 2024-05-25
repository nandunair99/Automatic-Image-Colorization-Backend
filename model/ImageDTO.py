from pydantic import BaseModel


class ImageDTO(BaseModel):
    base64_string: str
    type: str | None = None
