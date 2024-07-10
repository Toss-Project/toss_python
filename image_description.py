# STEP 1
from fastapi import FastAPI, File, UploadFile
import ollama
import base64 #표준 라이브러리에 있음

# STEP 2

app = FastAPI()

@app.post("/image_description/")
async def simulate_image_description(file: UploadFile):
    
    
    contents = await file.read()
    
    # 이미지를 base64로 인코딩
    image_base64 = base64.b64encode(contents).decode('utf-8')
    
    model = ollama.Client()
    
    # Bakllava 모델에 이미지와 프롬프트 전송
    prompt = "Please describe this image with different content in English three times within 30 characters in one template sentence. And don't say anything other than the three template sentences. Organize the three template sentences into numbers 1, 2, and 3, and just write the image description."
    response = model.generate(
        model='llava:7b',
        prompt=prompt,
        images=[image_base64]
    )
    # response2 = model.generate(
    #     model='llava:7b',
    #     prompt=prompt,
    #     images=[image_base64]
    # )
    # response3 = model.generate(
    #     model='llava:7b',
    #     prompt=prompt,
    #     images=[image_base64]
    # )
    # for i in range(3):
    #     if response1['response'] == response2['response']:
    #         response2 = await model.generate(
    #         model='bakllava',
    #         prompt=prompt,
    #         images=[image_base64]
    #         )
    
    # response_total=[response1['response'],response2['response'],response3['response']]
    # return response_total

    return response['response']

# 이미지 설명을 시뮬레이션합니다 (실제로는 이미지 분석 모델의 출력일 것입니다)
# image_description = "A red apple sitting on a wooden table next to an open book."

# result = simulate_image_description(image_description)


# print(result)