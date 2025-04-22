import os
import sys
import torch
from diffusers import AudioLDMPipeline, UNet2DConditionModel
from safetensors.torch import load_file
import soundfile as sf

def main():
    # 기본 모델 로드
    base_model_id = "cvssp/audioldm-s-full-v2"
    pipe = AudioLDMPipeline.from_pretrained(base_model_id)
    
    # LoRA 가중치 경로
    lora_weights_path = "/home/2020112030/WorkSpace/2025/AudioLDM-with-LoRA/script/train/AudioLDM-with-LoRA/data/LoRA_weight/checkpoint-50000/model.safetensors"
    
    # LoRA 가중치 로드
    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    state_dict = load_file(lora_weights_path)
    unet.load_state_dict(state_dict, strict=False)
    
    # 수정된 UNet을 파이프라인에 적용
    pipe.unet = unet
    
    # GPU로 이동
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    # 생성할 프롬프트 설정
    prompt = "The low quality recording features a funky blues acoustic guitar melody. The recording is in mono, a bit noisy, reverberant - as it was probably recorded with a phone and it sounds passionate. There are also some strings creaking sounds, while the player is playing the instrument."
    
    # 오디오 생성
    audio = pipe(
        prompt=prompt,
        num_inference_steps=50,
        audio_length_in_s=10.0,
        guidance_scale=7.5,
    ).audios[0]
    
    # 결과 저장
    output_dir = "./AudioLDM-with-LoRA/generated_audio"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_audio.wav")
    sf.write(output_path, audio, samplerate=16000)
    
    print(f"Generated audio saved to: {output_path}")

if __name__ == "__main__":
    main() 