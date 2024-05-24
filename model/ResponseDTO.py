from fastapi import FastAPI
from pydantic import BaseModel

class ResponseModel(BaseModel):
    message: str