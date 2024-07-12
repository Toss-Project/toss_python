from diffusers import StableDiffusionPipeline
import torch
from fastapi import FastAPI , Form
# from fastapi import APIRouter , Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

app = FastAPI()

# router = APIRouter()

# CORS 설정
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.post("/text_to_cartgoryImage/")
async def text_to_cartgoryImage(text: str = Form(...)):

    prompt = text
    image = pipe(prompt).images[0]  
        
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    torch.cuda.empty_cache()

    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")