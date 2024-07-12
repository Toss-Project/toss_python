from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import ollama
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.post("/correct_grammar/")
async def correct_grammar(file: UploadFile = Form(...)):
    contents = await file.read()
    text = contents.decode('utf-8')

    model = ollama.Client()
    
    # 텍스트 문법 교정을 위한 프롬프트 전송
    prompt = f"Please correct the grammar of the following text. Only return the corrected sentence without any additional explanations: \"{text}\""
    response = model.generate(
        model='your-grammar-correction-model',
        prompt=prompt
    )

    torch.cuda.empty_cache()

    return JSONResponse(content={"original_text": text, "corrected_text": response['response']})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
