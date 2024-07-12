from fastapi import APIRouter
from random import randint
import ollama
import torch
from diffusers import StableDiffusionPipeline
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

router = APIRouter()

# # CORS 설정
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
#     "*",  # 모든 도메인 허용
# ]

# CORS 설정
# origins = [
#     "http://localhost",
#     "http://localhost:3000"
# ]


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@router.get("/random-image-generator/")
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


# STEP 1
from fastapi import File, UploadFile
import base64

@app.post("/image_description/")
async def simulate_image_description(file: UploadFile = File(...)):
    
    contents = await file.read()
    
    # 이미지를 base64로 인코딩
    image_base64 = base64.b64encode(contents).decode('utf-8')
    
    model = ollama.Client()
    
    # Bakllava 모델에 이미지와 프롬프트 전송
    prompt = "Please describe this image with different content in English three times within 30 characters in one template sentence. And don't say anything other than the three template sentences. Organize the three template sentences into numbers 1, 2, and 3, and just write the image description."
    response = model.generate(
        model='llava:7b',
        prompt=prompt,
        images=[image_base64]
    )

    return response['response']