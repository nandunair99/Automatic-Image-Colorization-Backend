from fastapi import APIRouter, Depends, HTTPException

from colorizers.colorizer import Colorizer
from model.ResponseDTO import ResponseDTO
from model.ImageDTO import ImageDTO

router = APIRouter(
    prefix="/transformer",
    tags=["transformer"],
    responses={404: {"description": "Not found"}},
)


@router.post("/upload", status_code=200, response_model=ResponseDTO)
async def upload_file(image: ImageDTO) -> ResponseDTO:
    c1 = Colorizer(image.greyscale_image)
    c1.colorize()
    # You can process the image further here
    responseDTO = ResponseDTO(
        message="Image parsed successfully",
        processed_image=image.greyscale_image,
        status_code=200
    )
    return responseDTO
