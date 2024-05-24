from fastapi import FastAPI
from pydantic import BaseModel


class Image(BaseModel):
    base64_string: str
    type: str | None = None

app = FastAPI()


@app.post("/image/")
async def create_item(image: Image):
    return image