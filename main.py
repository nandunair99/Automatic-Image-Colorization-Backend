from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from routers import transformer
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.include_router(transformer.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8081)
