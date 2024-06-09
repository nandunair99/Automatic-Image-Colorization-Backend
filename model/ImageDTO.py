from pydantic import BaseModel


class ImageDTO(BaseModel):
    greyscale_image: str
