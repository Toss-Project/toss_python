from fastapi import FastAPI, File, UploadFile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import io
import torch
import librosa
import numpy as np

app = FastAPI()

# 모델과 프로세서 로드
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

TARGET_SR = 16000

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(..., max_size=1024*1024*10)):
    # 업로드된 파일에서 오디오 데이터 읽기
    contents = await file.read()
    
    # librosa를 사용하여 오디오 파일 로드 및 리샘플링
    audio, _ = librosa.load(io.BytesIO(contents), sr=TARGET_SR)
    
    # float32로 변환 및 정규화
    audio = librosa.util.normalize(audio.astype(np.float32))
    
    # 입력 특성 생성
    input_features = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features 

    # 영어 텍스트 생성
    with torch.no_grad():
        predicted_ids_en = model.generate(input_features, task="transcribe", language="en")
    
    # 한국어 번역 생성
    # with torch.no_grad():
    #     predicted_ids_ko = model.generate(input_features, task="translate", language="ko")
    
    # 토큰 ID를 텍스트로 디코딩
    transcription_en = processor.batch_decode(predicted_ids_en, skip_special_tokens=True)[0]

    return {"transcription_en": transcription_en}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)