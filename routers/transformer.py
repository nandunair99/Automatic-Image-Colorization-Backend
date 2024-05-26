from fastapi import APIRouter, Depends, HTTPException
from model.ResponseDTO import ResponseDTO
from model.ImageDTO import ImageDTO
import base64
from io import BytesIO
from PIL import Image  # for image processing (optional)

router = APIRouter(
    prefix="/transformer",
    tags=["transformer"],
    responses={404: {"description": "Not found"}},
)


@router.post("/upload", status_code=200, response_model=ResponseDTO)
async def upload_file(image: ImageDTO) -> ResponseDTO:
    print(image.base64_string)
    decoded_bytes = base64.b64decode(image.base64_string)
    decoded_io = BytesIO(decoded_bytes)
    img = Image.open(decoded_io)  # Assuming image data is encoded correctly
    # You can process the image further here
    responseDTO = ResponseDTO(
        message="Image parsed successfully",
        content=image.base64_string,
        status_code = 200
    )
    return responseDTO
