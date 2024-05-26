from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from routers import transformer

app = FastAPI()

app.include_router(transformer.router)

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8081)
