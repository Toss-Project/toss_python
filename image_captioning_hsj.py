# from fastapi import FastAPI, UploadFile, Form, File
# from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
# import torch
# from PIL import Image
# import io

# # 추론기 1
# model1 = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model1.to(device)

# max_length = 16
# num_beams = 4
# gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# def predict_step(images):
#     pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
#     pixel_values = pixel_values.to(device)

#     output_ids = model1.generate(pixel_values, **gen_kwargs)

#     preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#     preds = [pred.strip() for pred in preds]

#     return preds

# # 추론기 2
# processor2 = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model2 = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# # 추론기 3
# processor3 = AutoProcessor.from_pretrained("microsoft/git-base-coco")
# model3 = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

# app = FastAPI()

# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile):
#     content = await file.read()

#     # 바이트를 PIL 이미지로 변환
#     pil_img = Image.open(io.BytesIO(content)).convert("RGB")

#     # 추론기 1
#     result1 = predict_step([pil_img])

#     # 추론기 2
#     inputs = processor2(pil_img, return_tensors="pt")
#     out = model2.generate(**inputs)
#     result2 = processor2.decode(out[0], skip_special_tokens=True)

#     # 추론기 3
#     pixel_values = processor3(images=pil_img, return_tensors="pt").pixel_values
#     generated_ids = model3.generate(pixel_values=pixel_values, max_length=50)
#     result3 = processor3.batch_decode(generated_ids, skip_special_tokens=True)[0]

#     print("result 1 : " + str(result1))
#     print("result 2 : " + str(result2))
#     print("result 3 : " + str(result3))

#     # 합치기
#     result1.append(result2)
#     result1.append(result3)

#     # 분류 결과 처리
#     return result1

# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile):
#     content = await file.read()

#     # 바이트를 PIL 이미지로 변환
#     pil_img = Image.open(io.BytesIO(content)).convert("RGB")

#     # 추론기 1
#     result1 = predict_step([pil_img])

#     # 추론기 2
#     inputs = processor2(pil_img, return_tensors="pt")
#     out = model2.generate(**inputs)
#     result2 = processor2.decode(out[0], skip_special_tokens=True)

#     # 추론기 3
#     pixel_values = processor3(images=pil_img, return_tensors="pt").pixel_values
#     generated_ids = model3.generate(pixel_values=pixel_values, max_length=50)
#     result3 = processor3.batch_decode(generated_ids, skip_special_tokens=True)[0]

#     print("result 1 : " + str(result1))
#     print("result 2 : " + str(result2))
#     print("result 3 : " + str(result3))

#     # 합치기
#     result1.append(result2)
#     result1.append(result3)

#     # 분류 결과 처리
#     return result1

from fastapi import FastAPI
from random import randint
import ollama
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image
from io import BytesIO
from fastapi.responses import JSONResponse
import base64

model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

app = FastAPI()

@app.get("/random-image-generator/")
async def random_image_generator():

    # 랜덤 카테고리 선정
    random_category = ["fruit", "school", "hospital", "park", "mart", "friend", "football", "home", "airplane", "bus"]
    random_num = randint(0, 9)
    category = random_category[random_num]

    print("category : " + category)

    # ollama를 통해 문구 생성
    model = ollama.Client()
    prompt = "Write a descriptive sentence containing " + category + " within 30 characters."
    response = model.generate(
        model='llava:7b',
        prompt=prompt
    )

    generated_text = response['response']
    print("response : " + generated_text)

    image = pipe(generated_text).images[0]

    # 바이트 배열로 이미지 저장
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    print("img_str : " + img_str)

    return JSONResponse(content={'image': img_str})

