from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from random import randint
import ollama
import torch
import soundfile as sf
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import io
import numpy as np
import os

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

app = FastAPI()

#CORS 설정
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

# 모델과 프로세서 로드 (앱 시작 시 한 번만 로드)
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# 스피커 임베딩 로드
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

class TextRequest(BaseModel):
    text: str

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


@app.post("/image-description/")
async def image_description(file: UploadFile = File(...)):
    
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



@app.post("/text-to-speech/")
async def text_to_speech(request: TextRequest):
    try:
        # 텍스트를 입력 형식으로 변환
        inputs = processor(text=request.text, return_tensors="pt")

        # 음성 생성
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

        print(request.text)

        # NumPy 배열을 WAV 형식의 바이트로 변환
        byte_io = io.BytesIO()
        sf.write(byte_io, speech.numpy(), samplerate=16000, format='WAV')
        byte_io.seek(0)

        # 스트리밍 응답으로 오디오 데이터 반환
        return StreamingResponse(byte_io, media_type="audio/webm")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))