from fastapi import FastAPI, UploadFile
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
import io

# 추론기 1
model1 = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(images):
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model1.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds

# 추론기 2
processor2 = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model2 = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 추론기 3
processor3 = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model3 = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    content = await file.read()

    # 바이트를 PIL 이미지로 변환
    pil_img = Image.open(io.BytesIO(content)).convert("RGB")

    # 추론기 1
    result1 = predict_step([pil_img])

    # 추론기 2
    inputs = processor2(pil_img, return_tensors="pt")
    out = model2.generate(**inputs)
    result2 = processor2.decode(out[0], skip_special_tokens=True)

    # 추론기 3
    pixel_values = processor3(images=pil_img, return_tensors="pt").pixel_values
    generated_ids = model3.generate(pixel_values=pixel_values, max_length=50)
    result3 = processor3.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("result 1 : " + str(result1))
    print("result 2 : " + str(result2))
    print("result 3 : " + str(result3))

    # 합치기
    result1.append(result2)
    result1.append(result3)

    # 분류 결과 처리
    return result1
