import nemo.collections.asr as nemo_asr
import nemo.collections.tts as nemo_tts
import torchaudio

# ASR 모델 초기화
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="stt_en_jasper10x5dr")

# TTS 모델 초기화
tts_model = nemo_tts.models.FastPitchMelGANModel.from_pretrained(model_name="tts_en_fastpitch")

# 음성 파일 경로
audio_file = "path_to_your_audio_file.wav"

# 음성 파일을 텍스트로 변환 (ASR)
def transcribe_audio(audio_file):
    audio, sample_rate = torchaudio.load(audio_file)
    text = asr_model.transcribe([audio.numpy()], [sample_rate])
    return text

# 텍스트를 음성 파일로 변환 (TTS)
def synthesize_text(text):
    with torch.no_grad():
        spectrogram = tts_model.generate_spectrogram(text)
        audio = tts_model.convert_spectrogram_to_audio(spectrogram)
    return audio

# 음성 파일을 텍스트로 변환하여 출력
transcribed_text = transcribe_audio(audio_file)
print("Transcribed text:", transcribed_text)

# 텍스트를 음성으로 변환하여 파일 저장
output_audio = synthesize_text(transcribed_text)
output_audio_file = "output_audio.wav"
torchaudio.save(output_audio_file, output_audio)

print(f"Output audio saved to: {output_audio_file}")

