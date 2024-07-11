from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from happytransformer import HappyTextToText, TTSettings
import uvicorn

app = FastAPI()

# HappyTextToText 모델 초기화
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
args = TTSettings(num_beams=10, max_length=1000, min_length=1)

@app.post("/correct_grammar")
async def correct_grammar(file: UploadFile = File(...)):
    try:
        # 업로드된 파일 내용 읽기
        contents = await file.read()
        text = contents.decode("utf-8")

        # 문법 교정
        input_text = f"grammar: {text}"
        result = happy_tt.generate_text(input_text, args=args)

        # <red> 태그를 추가하여 문법 교정된 부분 강조
        corrected_text = result.text.replace("<red>", "<span class='correction-red'>").replace("</red>", "</span>")

        return JSONResponse(content={
            "original_text": text,
            "corrected_text": corrected_text,
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
