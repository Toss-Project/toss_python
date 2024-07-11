from fastapi import FastAPI
from random import randint
import ollama
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# model_id = "stabilityai/stable-diffusion-2"
# scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
# pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

app = FastAPI()

# CORS 설정
origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/random-image-generator/")
async def random_image_generator():
    # 랜덤 카테고리 선정
    random_category = ["fruit", "school", "hospital", "park", "mart", "friend", "football", "home", "airplane", "bus"]
    random_num = randint(0, 9)
    category = random_category[random_num]

    print("category : " + category)

    # ollama를 통해 문구 생성
    model = ollama.Client()
    prompt = "Write a descriptive sentence containing " + category + " within 30 characters."
    response = model.generate(
        model='llava:7b',
        prompt=prompt
    )

    generated_text = response['response']
    print("response : " + generated_text)

    image = pipe(generated_text).images[0]

    # 임시 파일에 이미지 저장
    temp_image_path = "/toss_python/image/random_image.png"
    image.save(temp_image_path)

    return FileResponse(temp_image_path, media_type="image/png", filename="generated_image.png")