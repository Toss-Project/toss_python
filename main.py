# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# import importlib
# import os
# import time

# app = FastAPI()

# # CORS 설정
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "PUT", "DELETE"],
#     allow_headers=["*"],
# )

# # 현재 디렉토리의 모든 .py 파일을 찾아서 임포트
# for filename in os.listdir("."):
#     if filename.endswith(".py") and filename != "main.py":
#         module_name = filename[:-3]  # .py 확장자 제거
#         module = importlib.import_module(module_name)
#         if hasattr(module, 'router'):
#             app.include_router(module.router)

# # 모든 요청에 대해 처리 시간을 헤더에 추가하는 미들웨어 추가
# @app.middleware("http")
# async def add_process_time_header(request: Request, call_next):
#     start_time = time.time()
#     response = await call_next(request)
#     process_time = time.time() - start_time
#     response.headers["X-Process-Time"] = str(process_time)
#     return response

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main:app",
#         host="127.0.0.1",
#         port=8000,
#         reload=True,
#         workers=4,
#         loop="uvloop",
#         http="httptools"
#     )