import os
import tempfile
import subprocess
import soundfile as sf
import noisereduce as nr
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pydub import AudioSegment
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import traceback
import ollama

# FastAPI application
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

# FFmpeg 경로 설정 (실제 경로로 변경해주세요)
ffmpeg_path = r"C:\\ffmpeg-2024-07-07-git-0619138639-full_build\\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

# 모델과 프로세서 로드
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# FFmpeg 경로 설정
AudioSegment.converter = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffmpeg = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(ffmpeg_path, "ffprobe.exe")

# 임시 디렉토리 생성 및 관리
temp_dir = tempfile.TemporaryDirectory()

# Ollama Client 초기화
ollama_client = ollama.Client()

# 노이즈 제거 함수
def reduce_noise(input_file, output_file):
    data, rate = sf.read(input_file)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    sf.write(output_file, reduced_noise, rate)

# ASR 엔드포인트
@app.post("/api/automaticspeechrecognition")
async def transcribe_audio(file: UploadFile = File(..., max_size=1024*1024*10)):  # 10MB 제한
    try:
        # 파일을 디스크에 저장
        audio_data = await file.read()
        audio_file = os.path.join(temp_dir.name, file.filename)
        with open(audio_file, "wb") as f:
            f.write(audio_data)

        # WEBM 파일을 WAV 파일로 변환
        wav_file = os.path.join(temp_dir.name, "converted.wav")
        subprocess.call([AudioSegment.ffmpeg, '-i', audio_file, wav_file])

        # 노이즈 제거 적용
        denoised_wav_file = os.path.join(temp_dir.name, "denoised.wav")
        reduce_noise(wav_file, denoised_wav_file)

        # WAV 파일 로드 및 처리
        waveform, sample_rate = torchaudio.load(denoised_wav_file)
        # (나머지 코드는 여기서부터)

    except Exception as e:
        print(f"Error occurred during ASR processing: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# 노이즈 제거 엔드포인트
@app.post("/denoise")
async def denoise_audio(file: UploadFile = File(..., max_size=1024*1024*10)):  # 10MB 제한
    try:
        # 파일을 디스크에 저장
        audio_data = await file.read()
        input_file = os.path.join(temp_dir.name, file.filename)
        with open(input_file, "wb") as f:
            f.write(audio_data)

        # 노이즈 제거 적용
        output_file = os.path.join(temp_dir.name, "denoised_" + file.filename)
        reduce_noise(input_file, output_file)

        # 처리된 파일 반환
        if not os.path.exists(output_file):
            raise HTTPException(status_code=500, detail="Denoised file was not created")

        return FileResponse(output_file, media_type="audio/wav", filename="denoised_audio.wav")

    except Exception as e:
        print(f"Error occurred during noise reduction: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
