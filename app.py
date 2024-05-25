from fastapi import FastAPI
from pydantic import BaseModel
from routes import item_routes

app = FastAPI()

app.include_router(item_routes.router)
