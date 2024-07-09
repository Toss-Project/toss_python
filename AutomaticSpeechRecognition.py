# import torchaudio
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# import torch

# def transcribe_audio(audio_file_path, model_name="facebook/wav2vec2-base-960h"):
#     # 모델과 프로세서 로드
#     processor = Wav2Vec2Processor.from_pretrained(model_name)
#     model = Wav2Vec2ForCTC.from_pretrained(model_name)

#     # 음성 파일 로드 및 변환
#     waveform, sample_rate = torchaudio.load(audio_file_path)

#     # 음성 파일이 16kHz가 아니라면 리샘플링
#     if sample_rate != 16000:
#         resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
#         waveform = resampler(waveform)
#         sample_rate = 16000

#     # 모델에 입력할 준비
#     input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=sample_rate).input_values

#     # 모델을 사용하여 텍스트 예측
#     with torch.no_grad():
#         logits = model(input_values).logits

#     # 예측 결과에서 텍스트 추출
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.batch_decode(predicted_ids)

#     return transcription[0]


# # 사용 예제
# audio_file_path = "sample1.wav"

# transcription = transcribe_audio(audio_file_path)
# print("Transcription: ", transcription)

# # 텍스트 파일로 저장
# with open("transcription.txt", "w") as f:
#     f.write(transcription)
from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

app = FastAPI()

# 모델과 프로세서 로드
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # 음성 파일 로드 및 변환
        waveform, sample_rate = torchaudio.load(file.file)

        # 음성 파일이 16kHz가 아니라면 리샘플링
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # 모델에 입력할 준비
        input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=sample_rate).input_values

        # 모델을 사용하여 텍스트 예측
        with torch.no_grad():
            logits = model(input_values).logits

        # 예측 결과에서 텍스트 추출
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)

        return {"transcription": transcription[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
