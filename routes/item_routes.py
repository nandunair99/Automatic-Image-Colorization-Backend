from typing import Any
from fastapi import APIRouter
from model import ImageDTO,ResponseDTO
import base64
from io import BytesIO
from PIL import Image

router = APIRouter()

@router.post("/file/upload2", response_model=ResponseDTO)
async def upload_file(image: ImageDTO) -> ResponseDTO:
    try:
        print(image.base64_string)
        decoded_bytes = base64.b64decode(image.base64_string)
        decoded_io = BytesIO(decoded_bytes)
        img = Image.open(decoded_io)  # Assuming image data is encoded correctly
        # You can process the image further here
        responseDTO = ResponseDTO(
            message=image.base64_string
        )
        return responseDTO
    except Exception as e:
        response = ResponseDTO(
            message=f"Failed to decode image: {e}"
        )
        return response
