from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import importlib
import os

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

# 현재 디렉토리의 모든 .py 파일을 찾아서 임포트
for filename in os.listdir("."):
    if filename.endswith(".py") and filename != "main.py":
        module_name = filename[:-3]  # .py 확장자 제거
        module = importlib.import_module(module_name)
        if hasattr(module, 'router'):
            app.include_router(module.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)