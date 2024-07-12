# STEP 1
import os
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import ollama
import base64 #표준 라이브러리에 있음
import torch
import torchaudio
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import subprocess
from fastapi.responses import StreamingResponse
import io
from diffusers import StableDiffusionPipeline

# FFmpeg 경로 설정 (실제 경로로 변경해주세요)
# 시스템 환경변수 -> 시스템변수-> path 추가해야함(아래 경로 추가)
ffmpeg_path = r"C:\\ffmpeg-2024-07-07-git-0619138639-full_build\\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path


app = FastAPI()

# CORS 설정
origins = [
    "*",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 모델과 프로세서 로드
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# 'tmp' 디렉토리 생성
tmp_dir = "tmp"
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

# FFmpeg 경로 직접 지정
AudioSegment.converter = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffmpeg = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(ffmpeg_path, "ffprobe.exe")

@app.post("/image-description")
async def simulate_image_description(file: UploadFile = File(...)):
    
    
    contents = await file.read()
    
    # 이미지를 base64로 인코딩
    image_base64 = base64.b64encode(contents).decode('utf-8')
    
    model = ollama.Client()
    
    # llava 모델에 이미지와 프롬프트 전송
    prompt = "Please describe this image with different content in English three times within 30 characters in one template sentence. And don't say anything other than the three template sentences. Organize the three template sentences into numbers 1, 2, and 3, and just write the image description."
    response = model.generate(
        model='llava:7b',
        prompt=prompt,
        images=[image_base64]
    )

    torch.cuda.empty_cache()

    return response['response']

@app.post("/image-text matching")
async def simulate_image_description(file: UploadFile = File(...),text: str = Form(...)):
    
    
    contents = await file.read()
    
    # 이미지를 base64로 인코딩
    image_base64 = base64.b64encode(contents).decode('utf-8')
    
    model = ollama.Client()
    
    # llava 모델에 이미지와 프롬프트 전송
    prompt = f"{text} Give us a score from 0 to 100 on how well this text describes the image."
    response = model.generate(
        model='llava:7b',
        prompt=prompt,
        images=[image_base64]
    )

    torch.cuda.empty_cache()

    return response['response']


@app.post("/text_to_cartgoryImage/")
async def text_to_cartgoryImage(text: str = Form(...)):

    prompt = text
    image = pipe(prompt).images[0]  
        
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    torch.cuda.empty_cache()

    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")


# @app.post("/api/automaticspeechrecognition")
@app.post("/api/automaticspeechrecognition")
async def transcribe_audio(file: UploadFile = File(..., max_size=1024*1024*10)):  # 10MB로 제한
    try:
        print(f"Received file: {file.filename}")
        print(f"File content type: {file.content_type}")
        
        # 파일을 디스크에 저장
        audio_data = await file.read()
        audio_file = os.path.join(tmp_dir, file.filename)
        with open(audio_file, "wb") as f:
            f.write(audio_data)
        
        print(f"File size: {len(audio_data)} bytes")
        
        # WEBM 파일을 WAV 파일로 변환
        wav_file = os.path.join(tmp_dir, "converted.wav")
        subprocess.call([AudioSegment.ffmpeg, '-i', audio_file, wav_file])

        # WAV 파일 로드
        try:
            audio = AudioSegment.from_wav(wav_file)
        except Exception as e:
            print(f"Error loading audio file: {str(e)}")
            print(f"FFmpeg path: {AudioSegment.ffmpeg}")
            print(f"FFprobe path: {AudioSegment.ffprobe}")
            raise

        # torchaudio를 사용하여 WAV 파일 로드
        waveform, sample_rate = torchaudio.load(wav_file)
        print(f"Loaded waveform: {waveform.shape}, Sample rate: {sample_rate}")

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # 모노로 변환 (필요한 경우)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=sample_rate).input_values

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        
        text_file = os.path.join(tmp_dir, "transcription.txt")
        with open(text_file, "w") as f:
            f.write(transcription[0])

        # 임시 파일 삭제
        os.remove(audio_file)
        os.remove(wav_file)
        os.remove(text_file)

        return {"transcription": transcription[0]}
    

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)









# 이미지 설명을 시뮬레이션합니다 (실제로는 이미지 분석 모델의 출력일 것입니다)
# image_description = "A red apple sitting on a wooden table next to an open book."

# result = simulate_image_description(image_description)


# print(result)