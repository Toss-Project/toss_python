# STEP 1
import os
import traceback
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import ollama
import base64 #표준 라이브러리에 있음
import torch
import torchaudio
import torch.nn as nn
import librosa
import argparse
import numpy as np
import math
import warnings
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import subprocess
from fastapi.responses import StreamingResponse
import io
from diffusers import StableDiffusionPipeline
import soundfile as sf
import base64
from pydantic import BaseModel
from datasets import load_dataset
from fastapi.responses import JSONResponse, FileResponse
import noisereduce as nr

# FFmpeg 경로 설정 (실제 경로로 변경해주세요)
# 시스템 환경변수 -> 시스템변수-> path 추가해야함(아래 경로 추가)
ffmpeg_path = r"C:\\ffmpeg-2024-07-07-git-0619138639-full_build\\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path



app = FastAPI()

# CORS 설정
origins = [
    "*",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델과 프로세서 로드 (앱 시작 시 한 번만 로드)
processor2 = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model2 = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# 스피커 임베딩 로드
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

class TextRequest(BaseModel):
    text: str


# 모델과 프로세서 로드
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# 'tmp' 디렉토리 생성
tmp_dir = "tmp"
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

# 임시 디렉토리 생성 및 관리
temp_dir = tempfile.TemporaryDirectory()


# FFmpeg 경로 직접 지정
AudioSegment.converter = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffmpeg = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(ffmpeg_path, "ffprobe.exe")


# Ollama Client 초기화
ollama_client = ollama.Client()


# 노이즈 제거 함수
def reduce_noise(input_file, output_file):
    data, rate = sf.read(input_file)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    sf.write(output_file, reduced_noise, rate)

def get_sinusoid_encoding(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class GOPT(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, input_dim=84):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads) for i in range(depth)])

        self.pos_embed = nn.Parameter(torch.zeros(1, 55, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        self.in_proj = nn.Linear(self.input_dim, embed_dim)
        self.mlp_head_phn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        self.mlp_head_word1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        self.phn_proj = nn.Linear(40, embed_dim)

        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt4 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.cls_token5 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt5 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        trunc_normal_(self.cls_token1, std=.02)
        trunc_normal_(self.cls_token2, std=.02)
        trunc_normal_(self.cls_token3, std=.02)
        trunc_normal_(self.cls_token4, std=.02)
        trunc_normal_(self.cls_token5, std=.02)

    def forward(self, x, phn):
        B = x.shape[0]
        phn_one_hot = torch.nn.functional.one_hot(phn.long()+1, num_classes=40).float()
        phn_embed = self.phn_proj(phn_one_hot)

        if self.embed_dim != self.input_dim:
            x = self.in_proj(x)

        x = x + phn_embed

        cls_token1 = self.cls_token1.expand(B, -1, -1)
        cls_token2 = self.cls_token2.expand(B, -1, -1)
        cls_token3 = self.cls_token3.expand(B, -1, -1)
        cls_token4 = self.cls_token4.expand(B, -1, -1)
        cls_token5 = self.cls_token5.expand(B, -1, -1)

        x = torch.cat((cls_token1, cls_token2, cls_token3, cls_token4, cls_token5, x), dim=1)

        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        u1 = self.mlp_head_utt1(x[:, 0])
        u2 = self.mlp_head_utt2(x[:, 1])
        u3 = self.mlp_head_utt3(x[:, 2])
        u4 = self.mlp_head_utt4(x[:, 3])
        u5 = self.mlp_head_utt5(x[:, 4])

        p = self.mlp_head_phn(x[:, 5:])

        w1 = self.mlp_head_word1(x[:, 5:])
        w2 = self.mlp_head_word2(x[:, 5:])
        w3 = self.mlp_head_word3(x[:, 5:])
        return u1, u2, u3, u4, u5, p, w1, w2, w3

def extract_features(audio_file, feature_type='mfcc', n_mfcc=86):
    try:
        print(f"Loading audio file: {audio_file}")
        waveform, sample_rate = librosa.load(audio_file, sr=None)
        print(f"Audio file loaded. Sample rate: {sample_rate}, Duration: {len(waveform)/sample_rate:.2f} seconds")
        
        if feature_type == 'mfcc':
            features = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc)
        else:
            raise ValueError("Unsupported feature type")
        
        target_length = 50
        if features.shape[1] < target_length:
            features = np.pad(features, ((0, 0), (0, target_length - features.shape[1])))
        else:
            features = features[:, :target_length]
        
        return torch.FloatTensor(features).transpose(0, 1).unsqueeze(0)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        raise

def load_model(model_path, embed_dim, num_heads, depth, input_dim):
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = GOPT(embed_dim=embed_dim, num_heads=num_heads, depth=depth, input_dim=input_dim)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"Model file not found: {model_path}. Initializing a new model.")
        model = GOPT(embed_dim=embed_dim, num_heads=num_heads, depth=depth, input_dim=input_dim)
    model.eval()
    return model

def get_pronunciation_scores(model, audio_file, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    audio_features = extract_features(audio_file, n_mfcc=input_dim).to(device)
    
    phns = torch.randint(0, 39, (1, 50)).to(device)
    
    with torch.no_grad():
        u1, u2, u3, u4, u5, p, w1, w2, w3 = model(audio_features, phns)
    
    utterance_scores = {
        "accuracy": u1.item() * 5,
        "completeness": u2.item() * 5,
        "fluency": u3.item() * 5,
        "prosodic": u4.item() * 5,
        "total": u5.item() * 5
    }
    
    phone_scores = p.squeeze().cpu().numpy().tolist()
    word_scores = {
        "accuracy": w1.mean().item() * 5,
        "stress": w2.mean().item() * 5,
        "total": w3.mean().item() * 5
    }
    
    return utterance_scores, phone_scores, word_scores



@app.post("/pronunciatio-assessment/")
async def assess_pronunciation(file: UploadFile = File(..., max_size=1024*1024*10)):  # 10MB로 제한
    try:
        print(f"Received file: {file.filename}")
        print(f"File content type: {file.content_type}")
        
        # 파일을 디스크에 저장
        audio_data = await file.read()
        audio_file = os.path.join(tmp_dir, file.filename)
        with open(audio_file, "wb") as f:
            f.write(audio_data)
        
        print(f"File size: {len(audio_data)} bytes")
        
        # WEBM 파일을 WAV 파일로 변환
        wav_file = os.path.join(tmp_dir, "converted.wav")
        subprocess.call([AudioSegment.ffmpeg, '-i', audio_file, wav_file])

        # 모델 설정
        args = argparse.Namespace()
        args.exp_dir = "./exp/"
        args.embed_dim = 24
        args.goptheads = 1
        args.goptdepth = 3
        args.am = 'paiia'

        feat_dim = {'librispeech': 84, 'paiia': 86, 'paiib': 88}
        input_dim = feat_dim[args.am]

        model_path = os.path.join(args.exp_dir, "models", "best_audio_model.pth")
        
        model = load_model(model_path, args.embed_dim, args.goptheads, args.goptdepth, input_dim)
        utterance_scores, phone_scores, word_scores = get_pronunciation_scores(model, wav_file, input_dim)

        # 임시 파일 삭제
        os.remove(audio_file)
        os.remove(wav_file)

        return {
            "utterance_scores": utterance_scores,
            "phone_scores": phone_scores,
            "word_scores": word_scores
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/image-description")
async def simulate_image_description(file: UploadFile = File(...)):
    
    
    contents = await file.read()
    
    # 이미지를 base64로 인코딩
    image_base64 = base64.b64encode(contents).decode('utf-8')
    
    model = ollama.Client()
    
    # llava 모델에 이미지와 프롬프트 전송
    prompt = "Please describe this image with different content in English three times within 30 characters in one template sentence. And don't say anything other than the three template sentences. Organize the three template sentences into numbers 1, 2, and 3, and just write the image description."
    response = model.generate(
        model='llava:7b',
        prompt=prompt,
        images=[image_base64]
    )

    torch.cuda.empty_cache()

    return response['response']

@app.post("/image-text matching")
async def simulate_image_description(file: UploadFile = File(...),text: str = Form(...)):
    
    
    contents = await file.read()
    
    # 이미지를 base64로 인코딩
    image_base64 = base64.b64encode(contents).decode('utf-8')
    
    model = ollama.Client()
    
    # llava 모델에 이미지와 프롬프트 전송
    prompt = f"{text} Give us a score from 0 to 100 on how well this text describes the image."
    response = model.generate(
        model='llava:7b',
        prompt=prompt,
        images=[image_base64]
    )

    torch.cuda.empty_cache()

    return response['response']


@app.post("/text_to_cartgoryImage/")
async def text_to_cartgoryImage(text: str = Form(...)):

    prompt = text
    image = pipe(prompt).images[0]  
        
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    torch.cuda.empty_cache()

    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")


# @app.post("/api/automaticspeechrecognition")
@app.post("/api/automaticspeechrecognition")
async def transcribe_audio(file: UploadFile = File(..., max_size=1024*1024*10)):  # 10MB로 제한
    try:
        print(f"Received file: {file.filename}")
        print(f"File content type: {file.content_type}")
        
        # 파일을 디스크에 저장
        audio_data = await file.read()
        audio_file = os.path.join(tmp_dir, file.filename)
        with open(audio_file, "wb") as f:
            f.write(audio_data)
        
        print(f"File size: {len(audio_data)} bytes")

        
        
        # WEBM 파일을 WAV 파일로 변환
        wav_file = os.path.join(tmp_dir, "converted.wav")
        subprocess.call([AudioSegment.ffmpeg, '-i', audio_file, wav_file])


        # 노이즈 제거 적용
        output_file = os.path.join(temp_dir.name, "converted.wav")
        reduce_noise(wav_file, output_file)

        # 처리된 파일 반환
        if not os.path.exists(output_file):
            raise HTTPException(status_code=500, detail="Denoised file was not created")

        # return FileResponse(output_file, media_type="audio/wav", filename="denoised_audio.wav")


        # WAV 파일 로드
        try:
            audio = AudioSegment.from_wav(wav_file)
        except Exception as e:
            print(f"Error loading audio file: {str(e)}")
            print(f"FFmpeg path: {AudioSegment.ffmpeg}")
            print(f"FFprobe path: {AudioSegment.ffprobe}")
            raise

        # torchaudio를 사용하여 WAV 파일 로드
        waveform, sample_rate = torchaudio.load(wav_file)
        print(f"Loaded waveform: {waveform.shape}, Sample rate: {sample_rate}")

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # 모노로 변환 (필요한 경우)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=sample_rate).input_values

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        
        text_file = os.path.join(tmp_dir, "transcription.txt")
        with open(text_file, "w") as f:
            f.write(transcription[0])

        # 임시 파일 삭제
        os.remove(audio_file)
        os.remove(wav_file)
        os.remove(text_file)

        return {"transcription": transcription[0]}
    

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
# 노이즈 제거 엔드포인트
# @app.post("/denoise")
# async def denoise_audio(file: UploadFile = File(..., max_size=1024*1024*10)):  # 10MB 제한
#     try:
#         # 파일을 디스크에 저장
#         audio_data = await file.read()
#         input_file = os.path.join(temp_dir.name, file.filename)
#         with open(input_file, "wb") as f:
#             f.write(audio_data)

#         # 노이즈 제거 적용
#         output_file = os.path.join(temp_dir.name, "denoised_" + file.filename)
#         reduce_noise(input_file, output_file)

#         # 처리된 파일 반환
#         if not os.path.exists(output_file):
#             raise HTTPException(status_code=500, detail="Denoised file was not created")

#         return FileResponse(output_file, media_type="audio/wav", filename="denoised_audio.wav")

#     except Exception as e:
#         print(f"Error occurred during noise reduction: {str(e)}")
#         print(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/text-to-speech/")
async def text_to_speech(request: TextRequest):
    try:
        print(request.text)
        # 텍스트를 입력 형식으로 변환
        inputs = processor2(text=request.text, return_tensors="pt")

        # 음성 생성
        speech = model2.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)


        # NumPy 배열을 WAV 형식의 바이트로 변환
        byte_io = io.BytesIO()
        sf.write(byte_io, speech.numpy(), samplerate=16000, format='WAV')
        byte_io.seek(0)

        # 스트리밍 응답으로 오디오 데이터 반환
        return StreamingResponse(byte_io, media_type="audio/webm")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Grammar Correction using Ollama
@app.post("/correct-grammar/")
async def correct_grammar(text: TextRequest):
    
    try:
        prompt = f"Please correct the grammar of the following text. Only return the corrected sentence without any additional explanations: \"{text}\""
        response = ollama_client.generate(
            model='gemma2:latest',
            prompt=prompt
        )

        corrected_text = response['response']
        print(corrected_text)
        return {
            "original_text": text,
            "corrected_text": corrected_text,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)









# 이미지 설명을 시뮬레이션합니다 (실제로는 이미지 분석 모델의 출력일 것입니다)
# image_description = "A red apple sitting on a wooden table next to an open book."

# result = simulate_image_description(image_description)


# print(result)