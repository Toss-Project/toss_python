from fastapi import FastAPI, UploadFile, File, HTTPException
import speech_recognition as sr
from pydub import AudioSegment

app = FastAPI()

# 음성 인식기 객체 생성
recognizer = sr.Recognizer()

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # 업로드된 음성 파일 저장
        file_contents = await file.read()
        with open("uploaded_audio.wav", "wb") as f:
            f.write(file_contents)

        # 음성 파일 읽기
        with sr.AudioFile("uploaded_audio.wav") as source:
            audio_data = recognizer.record(source)

            # Google Web Speech API를 이용하여 음성 인식
            text = recognizer.recognize_google(audio_data, language="en-US")
            print("인식된 텍스트:", text)

            # 인식된 텍스트를 파일로 저장
            text_file = "transcription.txt"
            with open(text_file, "w") as f:
                f.write(text)

            return {"transcription": text}

    except sr.UnknownValueError:
        raise HTTPException(status_code=500, detail="음성을 인식할 수 없습니다.")

    except sr.RequestError as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))