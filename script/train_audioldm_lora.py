'''
Copyright Dongguk U.V. CSE 2020112030 KIM SEON PYO [2025/03/04 ~]

해당 코드는 DiffusersPipeline의 AudioLDM 모델을 파인튜닝하는 LoRA 가중치를 학습하는 코드입니다.
README.md 를 참고해 주세요

This code is training LoRA weight for AudioLDM in DiffusersPipeline.
you can choose Base_model for training LoRA weight and save at [AudioLDM-with-LoRA/data/LoRA_weight/~]

adapt at app.py to use 
'''
import sys
import torch, random, torchaudio
import numpy as np
from tqdm.auto import tqdm
from diffusers import AudioLDMPipeline  # AudioLDM 파이프라인 로드
from peft import LoraConfig, get_peft_model  # LoRA 설정 및 모델 적용
import wandb  
from accelerate import Accelerator  # 모델 훈련 가속화
from accelerate.logging import get_logger  # 로그 출력
from accelerate.utils import ProjectConfiguration  
import torch.nn.functional as F
from transformers import AutoProcessor, ClapModel  # CLAP 모델 로드

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLAP 모델 및 프로세서 로드
processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)

# CLAP 유사도 - 오디오와 텍스트 임베딩의 코사인 유사도
def compute_clap_similarity(audio_waveform: np.ndarray, text: str) -> float:
    inputs = processor(audios=audio_waveform, return_tensors="pt", sampling_rate=48000)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        audio_embed = clap_model.get_audio_features(**inputs)
        text_embed = clap_model.get_text_features(**processor(text=text, return_tensors="pt", padding=True).to(device))
        audio_embed = F.normalize(audio_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        similarity = (audio_embed @ text_embed.T).item()
    return similarity

# 같은 실험값을 위한 seed
torch.manual_seed(42)

# 실험 설정값
default_model_id = "cvssp/audioldm-s-full-v2"
config = {
    "mode": "text_encoder",  # LoRA 적용 대상
    "target_modules": [
        f"text_model.encoder.layer.{i}.attention.self.query" for i in range(12)
    ] + [
        f"text_model.encoder.layer.{i}.attention.self.value" for i in range(12)
    ],
    "rank": 1,  # LoRA 랭크
    "alpha": 8,  # LoRA 스케일링 계수
    "learning_rate": 1e-4,
    "max_train_steps": 10, # 훈련 횟수수
    "prompt": "hiphop",  # 오디오 생성 프롬프트
    "name": "LoRA_WqWv_rank1"
}

# wandb 실험 추적 시작
run = wandb.init(
    project="AudioLDM-with-LoRA",
    name=config["name"],
    config=config,
)

# accelerate 설정
accelerator_project_config = ProjectConfiguration(project_dir="./", logging_dir="AudioLDM-with-LoRA/log")
accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision=None,
    log_with="wandb",
    project_config=accelerator_project_config,
)

logger = get_logger(__name__)
logger.info(accelerator.state, main_process_only=True)

# AudioLDM 모델 로드 
pipe = AudioLDMPipeline.from_pretrained(default_model_id)
pipe.to(accelerator.device)

# text_encoder에만 LoRA 적용
text_encoder = pipe.text_encoder
lora_config = LoraConfig(
    r=config["rank"],
    lora_alpha=config["alpha"],
    init_lora_weights="gaussian",
    target_modules=config["target_modules"]
)
text_encoder = get_peft_model(text_encoder, lora_config)

# unet, vae는 파인튜닝에서 제외 (freeze)
pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(False)
text_encoder.requires_grad_(True) 
# pretrained 모델 가중치는 requires_grad=False로 자동 설정 / 새롭게 삽입된 LoRA 레이어만 requires_grad=True로 설정되어 학습
text_encoder.to(accelerator.device, dtype=torch.float32)
pipe.text_encoder = text_encoder

# 옵티마이저 설정 (LoRA 파라미터만 대상)
optimizer = torch.optim.AdamW(
    params=filter(lambda p: p.requires_grad, text_encoder.parameters()),
    lr=config["learning_rate"]
)

# 학습
ref_path = "/home/2019111986/AudioLDM-with-LoRA/data/reference/hip-hop-trap-drums_155bpm.wav" # MSE loss 계산 refrence
clap_scores = []
losses = []

# max_train_steps만큼 오디오 생성,training, loss 계산, CLAP 유사도 계산
for step in range(config["max_train_steps"]):
    optimizer.zero_grad()

    result = pipe(config["prompt"], num_inference_steps=30) # num_inference_steps == Denoising step의 수
    audio = result["audios"][0]  
    audio_tensor = torch.tensor(audio, requires_grad=True).to(accelerator.device)

    ref_waveform, _ = torchaudio.load(ref_path)
    ref_waveform = ref_waveform[0, :audio_tensor.shape[0]].to(accelerator.device) # audio_tensor => requires_grad=True로 설정해 학습 가능하게게

    # 손실 계산 및 학습
    loss = F.mse_loss(audio_tensor, ref_waveform)
    accelerator.backward(loss) # BackPropagation
    optimizer.step() # Weight Update

    # CLAP 유사도 계산
    clap_score = compute_clap_similarity(audio, config["prompt"])

    clap_scores.append(clap_score)
    losses.append(loss.item())

    # wandb 로그
    wandb.log({
        "loss": loss.item(),
        "clap_score": clap_score,
        "audio": wandb.Audio(audio, sample_rate=16000, caption=f"{config['prompt']} step {step+1}")
    })

# 평균 clap_score, loss 출력
wandb.log({
    "average_clap_score": np.mean(clap_scores),
    "average_loss": np.mean(losses)
})
wandb.finish()
