from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from happytransformer import HappyTextToText, TTSettings
import uvicorn
import os
import traceback
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import subprocess

# FFmpeg 경로 설정 (실제 경로로 변경해주세요)
# 시스템 환경변수 -> 시스템변수-> path 추가해야함(아래 경로 추가)
ffmpeg_path = r"C:\\ffmpeg-2024-07-07-git-0619138639-full_build\\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

app = FastAPI()

# CORS 설정
origins = ["*"]

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

# 'tmp' 디렉토리 생성
tmp_dir = "tmp"
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

# FFmpeg 경로 직접 지정
AudioSegment.converter = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffmpeg = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(ffmpeg_path, "ffprobe.exe")


# HappyTextToText 모델 초기화
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
args = TTSettings(num_beams=10, max_length=1000, min_length=1)

class TextRequest(BaseModel):
    text: str

# ASR #############################################################################
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
# ASR/ #############################################################################

@app.post("/correct_grammar")
async def correct_grammar(request: TextRequest):
    try:
        # 업로드된 파일 내용 읽기
        # contents = await file.read()
        # text = contents.decode("utf-8")

        # 문법 교정
        input_text = f"grammar: {request.text}"
        result = happy_tt.generate_text(input_text, args=args)

        # <red> 태그를 추가하여 문법 교정된 부분 강조
        corrected_text = result.text.replace("<red>", "<span class='correction-red'>").replace("</red>", "</span>")

        return JSONResponse(content={
            "original_text": request.text,
            "corrected_text": corrected_text,
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
