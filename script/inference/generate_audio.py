import os
import torch
from diffusers import AudioLDMPipeline, UNet2DConditionModel
from diffusers import DiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
import soundfile as sf

def main():
    # 기본 설정
    base_model_id = "cvssp/audioldm-s-full-v2"
    lora_weights_path = "/home/2020112030/WorkSpace/2025/AudioLDM-with-LoRA/data/LoRA_weight/r2_alpha2/checkpoint-19400/model.safetensors"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. UNet 로드
    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet").to(device)

    # 2. LoRA 구성 설정
    lora_config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["to_q", "to_v"],
        init_lora_weights="gaussian"
    )

    # 3. UNet에 LoRA 구조 적용
    unet_lora = get_peft_model(unet, lora_config)

    # 4. LoRA 가중치 로드 및 적용
    lora_state = load_file(lora_weights_path)
    unet_lora.load_state_dict(lora_state, strict=False)

    # 5. PEFT 모델의 state_dict → diffusers 형식으로 변환
    lora_diffusers_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet_lora))

    # 6. 원래 UNet에 적용
    unet.load_state_dict(lora_diffusers_state, strict=False)

    # 7. AudioLDM 파이프라인 생성 및 수정된 UNet 적용
    pipe = DiffusionPipeline.from_pretrained(base_model_id, unet=unet).to(device)

    # 8. 프롬프트 설정 및 오디오 생성
    prompt = "An instrumental hip-hop track in the subgenre of boom bap, driven by punchy, repetitive kick drum patterns and groovy bass guitar lines. The drums remain present and consistent throughout the track, forming the backbone of the rhythm. The production emphasizes a raw, old-school vibe with minimal melodic elements, focusing on rhythm and head-nodding beats. The subgenre of hip-hop is boom bap."

    audio = pipe(
        prompt=prompt,
        num_inference_steps=50,
        audio_length_in_s=10.0,
        guidance_scale=5.0,
    ).audios[0]

    # 9. 결과 저장
    output_dir = "./generated_audio_LoRA" #/r2_alpha4/19400"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ex.wav")
    sf.write(output_path, audio, samplerate=16000)
    print(f"Generated audio saved to: {output_path}")

if __name__ == "__main__":
    main()
