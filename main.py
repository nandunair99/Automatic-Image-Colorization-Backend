from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import os
import base64
from io import BytesIO
from PIL import Image  # for image processing (optional)
from model import ResponseDTO, ImageDTO

# app = FastAPI()
#
#
# @app.get('/first')
# async def root():
#     return {
#         "example": "Created the first sample API", "data": "No data"
#     }
#
#
# IMAGEDIR = "image_store/"
#
#
# @app.post("/file/upload")
# async def upload_file(file: UploadFile = File(...)):
#     try:
#         # Ensure filename is safe
#         safe_filename = os.path.basename(file.filename)
#
#         # Construct the full path
#         full_path = os.path.join(IMAGEDIR, safe_filename)
#
#         # Read the file content
#         contents = await file.read()
#
#         # Save the file in binary write mode
#         with open(full_path, "wb") as f:
#             f.write(contents)
#
#         return {"filename": safe_filename, "message": "File saved successfully"}
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.post("/file/upload2", response_model=ResponseDTO)
# async def upload_file(image: ImageDTO) -> ResponseDTO:
#     try:
#         print(image.base64_string)
#         decoded_bytes = base64.b64decode(image.base64_string)
#         decoded_io = BytesIO(decoded_bytes)
#         img = Image.open(decoded_io)  # Assuming image data is encoded correctly
#         # You can process the image further here
#         responseDTO = ResponseDTO(
#             message=image.base64_string
#         )
#         return responseDTO
#     except Exception as e:
#         response = ResponseDTO(
#             message=f"Failed to decode image: {e}"
#         )
#         return response


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8081)
