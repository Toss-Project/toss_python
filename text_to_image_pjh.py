from diffusers import StableDiffusionPipeline
import torch
from fastapi import FastAPI, Form

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
app = FastAPI()


@app.post("/generate_image/")
async def generate_image_from_text(text: str = Form(...)):

    prompt = text
    image = pipe(prompt).images[0]  
        
    image_path = f"{text}.png"
    image.save(image_path)

    # return {"image_path": image_path}
    return image.show(image_path)