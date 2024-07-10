from fastapi import FastAPI, UploadFile, File
from sentence_transformers import SentenceTransformer, util
import numpy as np
import uvicorn

app = FastAPI()

# 모델 로드
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 텍스트 파일 읽기
def read_text_file(file_content):
    return file_content.decode("utf-8").splitlines()

@app.post("/compare-texts/")
async def compare_texts(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    file_content1 = await file1.read()
    file_content2 = await file2.read()
    
    sentences1 = read_text_file(file_content1)
    sentences2 = read_text_file(file_content2)

    # 문장 임베딩 생성
    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)

    # 유사도 계산
    similarities = np.zeros((len(sentences1), len(sentences2)))

    for i, embedding1 in enumerate(embeddings1):
        for j, embedding2 in enumerate(embeddings2):
            similarities[i][j] = util.pytorch_cos_sim(embedding1, embedding2).item()

    similarities_result = similarities * 100

    return {"similarities_result": similarities_result.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
