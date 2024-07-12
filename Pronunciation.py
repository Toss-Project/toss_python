import os
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import librosa
import numpy as np
import argparse
from pydub import AudioSegment
import subprocess
import math
import warnings

# FFmpeg 경로 설정 (실제 경로로 변경해주세요)
ffmpeg_path = r"C:\\ffmpeg-2024-07-07-git-0619138639-full_build\\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

app = FastAPI()

# CORS 설정
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 'tmp' 디렉토리 생성
tmp_dir = "tmp"
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

# FFmpeg 경로 직접 지정
AudioSegment.converter = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffmpeg = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(ffmpeg_path, "ffprobe.exe")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)