from fastapi import FastAPI, Form
# STEP 1
from transformers import pipeline


#STEP 2 : 모델을 자동으로 받아줌
# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model") #pipeline(감정 분석, 사용모델)
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC") 


app = FastAPI()


@app.post("/textclassification/")
async def textclassification(text: str = Form()):

    #STEP 3 
    # text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    # text = "샤오미의 폴더블 폰의 점유율이 삼성전자보다 높아졌다" #비정형 데이터: 문장을 보고 해석이 달라질 수 있음 
    # 비정형을 어떻게 정형화 시킬 지에 대해서 고민해 로직을 처리해야한다

    #STEP 4
    result = classifier(text)


    #STEP 5
    print(result)

    return {"result": result}