from fastapi import FastAPI, File, UploadFile, HTTPException
import json
import uvicorn
import uuid
import os

app = FastAPI()


@app.get('/first')
async def root():
    return {
        "example": "Created the first sample API", "data": "No data"
    }


IMAGEDIR = "image_store/"


@app.post("/file/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Ensure filename is safe
        safe_filename = os.path.basename(file.filename)

        # Construct the full path
        full_path = os.path.join(IMAGEDIR, safe_filename)

        # Read the file content
        contents = await file.read()

        # Save the file in binary write mode
        with open(full_path, "wb") as f:
            f.write(contents)

        return {"filename": safe_filename, "message": "File saved successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8081)
