# from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
# import torch
# import soundfile as sf
# from datasets import load_dataset

# def text_to_speech(input_file, output_file):
#     # 모델과 프로세서 로드
#     processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
#     model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
#     vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

#     # 스피커 임베딩 로드
#     embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
#     speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

#     # 텍스트 파일 읽기
#     with open(input_file, 'r', encoding='utf-8') as file:
#         text = file.read().strip()

#     # 텍스트를 입력 형식으로 변환
#     inputs = processor(text=text, return_tensors="pt")

#     # 음성 생성
#     speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

#     # 오디오 파일로 저장
#     sf.write(output_file, speech.numpy(), samplerate=16000)

#     print(f"음성 파일이 {output_file}에 저장되었습니다.")

# # 사용 예
# input_text_file = "transcription.txt"  # 여기에 실제 텍스트 파일 경로를 입력하세요
# output_audio_file = "output_audio.wav"

# text_to_speech(input_text_file, output_audio_file)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
from datasets import load_dataset
import os

app = FastAPI()

# 모델과 프로세서 로드 (앱 시작 시 한 번만 로드)
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# 스피커 임베딩 로드
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

@app.post("/text-to-speech/")
async def text_to_speech(file: UploadFile = File(...)):
    try:
        # 업로드된 파일 내용 읽기
        contents = await file.read()
        text = contents.decode("utf-8").strip()

        # 텍스트를 입력 형식으로 변환
        inputs = processor(text=text, return_tensors="pt")

        # 음성 생성
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

        # 오디오 파일로 저장
        output_file = f"output_{file.filename.split('.')[0]}.wav"
        sf.write(output_file, speech.numpy(), samplerate=16000)

        return JSONResponse(content={"message": f"음성 파일이 {output_file}에 저장되었습니다.", "file_path": output_file})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)