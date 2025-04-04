'''
Copyright Dongguk U.V. CSE 2020112030 KIM SEON PYO [2025/03/04 ~]

해당 코드는 DiffusersPipeline의 AudioLDM 모델을 파인튜닝하는 기법 중 하나인 LoRA에 관한 가중치를 학습하는 코드입니다.
README.md 를 참고해 주세요

This code is training LoRA weight for AudioLDM in DiffusersPipeline.
you can choose Base_model for training LoRA weight and save at [AudioLDM-with-LoRA/data/LoRA_weight/~]

adapt at app.py to use 
'''
import torch, random, torchaudio
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from diffusers import AudioLDMPipeline
from torchaudio import transforms as AT
from torchvision import transforms as IT
from peft import LoraConfig, get_peft_model
import wandb

# 실험 설정
torch.manual_seed(42)
base_model_id = "cvssp/audioldm-s-full-v2"

config = {
    "mode": "text_encoder",
    "target_modules": [
        f"text_model.encoder.layer.{i}.attention.self.query" for i in range(12)
    ] + [
        f"text_model.encoder.layer.{i}.attention.self.value" for i in range(12)
    ],
    "rank": 1,
    "alpha": 8,
    "learning_rate": 0.0001,
    "max_train_steps": 100,
    "base_model": base_model_id,
    "prompt": "hiphop",
    "name": "LoRA_WqWv_rank1"
}

def compute_snr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio (SNR) between reference and estimate signals"""
    reference = reference.view(-1)
    estimate = estimate.view(-1)

    noise = reference - estimate
    signal_power = torch.sum(reference ** 2)
    noise_power = torch.sum(noise ** 2) + 1e-9 

    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

# wandb 초기화
run = wandb.init(
    project="AudioLDM-with-LoRA",
    name=config["name"],
    config=config
)
# AudioLDM 모델 불러오기
pipe = AudioLDMPipeline.from_pretrained(config["base_model"])
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# text_encoder에 LoRA 적용
text_encoder = pipe.text_encoder

print("text_encoder 모듈 구조:")
for name, module in text_encoder.named_modules():
    print(name)

lora_config = LoraConfig(
    r=config["rank"],
    lora_alpha=config["alpha"],
    init_lora_weights="gaussian",
    target_modules=config["target_modules"]
)
text_encoder = get_peft_model(text_encoder, lora_config)
pipe.text_encoder = text_encoder

# 오디오 생성 및 wandb 로그
num_samples = 3
samples = []
for i in range(num_samples):
    result = pipe(config["prompt"], num_inference_steps=30)
    audio = result["audios"][0]  # numpy ndarray [T]
    audio_tensor = torch.tensor(audio)

    # 참고용 clean 오디오 불러오기 (길이 맞춰서)
    ref_waveform, _ = torchaudio.load("path_to_reference.wav")
    ref_waveform = ref_waveform[0, :audio_tensor.shape[0]]

    # SNR 계산
    snr_value = compute_snr(ref_waveform, audio_tensor)

    # wandb에 로그 저장
    wandb.log({
        f"sample_{i}_snr": snr_value,
        f"sample_{i}": wandb.Audio(audio, sample_rate=16000, caption=f"{config['prompt']} - {i}")
    })

# 최종 wandb 로그
wandb.log({
    "total_generated": num_samples,
    "average_snr": np.mean([compute_snr(ref_waveform, torch.tensor(a)) for a in samples])
})
wandb.finish()
