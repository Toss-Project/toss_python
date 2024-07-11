from diffusers import StableDiffusionPipeline
import torch
from fastapi import APIRouter , Form
from fastapi.responses import StreamingResponse
import io

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

router = APIRouter()


@router.post("/text_to_cartgoryImage/")
async def text_to_cartgoryImage(text: str = Form(...)):

    prompt = text
    image = pipe(prompt).images[0]  
        
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")