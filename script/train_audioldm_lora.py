import logging
import os
import numpy as np
import pandas as pd
import torch, torchaudio
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, ClapModel
from diffusers import AudioLDMPipeline
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from torch.utils.data import Dataset, DataLoader
import wandb

logger = get_logger(__name__)

# 같은 실험값을 위한 seed
torch.manual_seed(42)

# audioldm 모델
base_model_id = "cvssp/audioldm-s-full-v2"
config = {
    # unet 타켓 레이어
    "unet_target_modules": [
        f"down_blocks.{i}.attentions.{j}.transformer_blocks.0.{attn}.{proj}"
        for i in range(1, 4)
        for j in range(2)
        for attn in ['attn1', 'attn2']
        for proj in ['to_q', 'to_v']
    ] + [
        f"up_blocks.{i}.attentions.{j}.transformer_blocks.0.{attn}.{proj}"
        for i in range(3)
        for j in range(3)
        for attn in ['attn1', 'attn2']
        for proj in ['to_q', 'to_v']
    ] + [
        f"mid_block.attentions.0.transformer_blocks.0.{attn}.{proj}"
        for attn in ['attn1', 'attn2']
        for proj in ['to_q', 'to_v']
    ],
    # vae 타켓 레이어
    "vae_target_modules": [
        f"{i}.mid_block.attentions.0.{proj}"
        for i in ["encoder", "decoder"]
        for proj in ["to_q", "to_v"]
    ],
    "rank": 4,  # LoRA 랭크
    "alpha": 8,  # LoRA 스케일링 계수
    "learning_rate": 1e-4,
    "max_train_steps": 20, # 훈련 횟수
}

# wandb 설정
run = wandb.init(
    entity="",
    project="",
    config=config,
)

accelerator_project_config = ProjectConfiguration(project_dir="./", logging_dir="AudioLDM-with-LoRA/log")
accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision=None,
    log_with="wandb",
    project_config=accelerator_project_config,
)

# CLAP 모델 및 프로세서 로드
processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(accelerator.device)

# CLAP 유사도 - 오디오와 텍스트 임베딩의 코사인 유사도
def compute_clap_similarity(audio_waveform: np.ndarray, text: str) -> float:
    inputs = processor(audios=audio_waveform, return_tensors="pt", sampling_rate=48000)
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    with torch.no_grad():
        audio_embed = clap_model.get_audio_features(**inputs)
        text_embed = clap_model.get_text_features(**processor(text=text, return_tensors="pt", padding=True).to(accelerator.device))
        audio_embed = F.normalize(audio_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        similarity = (audio_embed @ text_embed.T).item()
        # [-1, 1] -> [0, 1]로 변환
        similarity = (similarity + 1) / 2
    return similarity

if torch.backends.mps.is_available():
    accelerator.native_amp = False

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)

# AudioLDM 파이프라인 로드
pipe = AudioLDMPipeline.from_pretrained(base_model_id)
pipe.to(accelerator.device)

# unet, vae만 LoRA 학습
pipe.unet.requires_grad_(True)
pipe.vae.requires_grad_(True)
pipe.text_encoder.requires_grad_(False)

pipe.unet.to(accelerator.device, dtype=torch.float32)
pipe.vae.to(accelerator.device, dtype=torch.float32)
pipe.text_encoder.to(accelerator.device, dtype=torch.float32)

# unet LoRA 설정
unet_lora_config = LoraConfig(
    r=config["rank"],
    lora_alpha=config["alpha"],
    init_lora_weights="gaussian",
    target_modules=config["unet_target_modules"],
)

pipe.unet = get_peft_model(pipe.unet, unet_lora_config)

# vae LoRA 설정
vae_lora_config = LoraConfig(
    r=config["rank"],
    lora_alpha=config["alpha"],
    init_lora_weights="gaussian",
    target_modules=config["vae_target_modules"],
)

pipe.vae = get_peft_model(pipe.vae, vae_lora_config)

# 옵티마이저 설정
unet_optimizer = torch.optim.AdamW(
    params=filter(lambda p: p.requires_grad, pipe.unet.parameters()),
    lr=config["learning_rate"]
)
vae_optimizer = torch.optim.AdamW(
    params=filter(lambda p: p.requires_grad, pipe.vae.parameters()),
    lr=config["learning_rate"]
)

# 데이터셋 샘플링(musiccaps를 로컬에 다운로드 및 해당 오디오를 다운로드하여 사용함)
class MusicCapsDataset(Dataset):
    def __init__(self, csv_path, audio_dir, max_samples=config["max_train_steps"], target_duration=10, sampling_rate=48000):
        # csv 파일의 max_samples만큼 사용
        self.data = pd.read_csv(csv_path).head(max_samples)
        self.audio_dir = audio_dir              # 오디오 파일 경로
        self.target_duration = target_duration  # 10초(오디오 길이)
        self.sampling_rate = sampling_rate      # 오디오 샘플링 레이트(초당 쪼개는 정도)
        self.target_length = self.target_duration * self.sampling_rate  # 오디오 샘플 수

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # idx에 해당하는 row 가져옴
        row = self.data.iloc[idx]
        # 유튜브 id, 캡션, 시작 시간 가져옴
        ytid = row["ytid"]
        caption = row["caption"]
        start_s = row.get("start_s", 0)

        audio_path = os.path.join(self.audio_dir, f"{ytid}.wav")
        # 오디오 파일 로드
        audio, sr = torchaudio.load(audio_path)
        audio = audio[0]  # mono 채널만

        # 샘플링 레이트가 다르면 리샘플링
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            audio = resampler(audio)

        # 오디오 길이 맞추기
        start_sample = int(start_s * self.sampling_rate)
        end_sample = start_sample + self.target_length

        # 오디오 길이가 target_length보다 짧으면 패딩
        if end_sample > audio.shape[0]:
            audio = torch.nn.functional.pad(audio, (0, end_sample - audio.shape[0]))

        # 시작 위치부터 10초까지 사용
        audio = audio[start_sample:end_sample]

        return audio, caption

# 데이터셋 경로
csv_path = "./musiccaps/musiccaps-public.csv"
audio_dir = "./musiccaps/audio"

dataset = MusicCapsDataset(csv_path, audio_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 학습
clap_scores = []
losses = []
step = 0
for audios, texts in dataloader:
    if step >= config["max_train_steps"]:
        break

    unet_optimizer.zero_grad()
    vae_optimizer.zero_grad()

    # 오디오와 텍스트 가져오기
    prompt = texts[0]
    ref_audio = audios[0].to(accelerator.device)

    # 오디오 생성(데이터셋의 caption을 사용)
    result = pipe(prompt, num_inference_steps=30, audio_length_in_s=10.0)

    # 오디오 데이터 가져와서 변환
    generated_audio = result["audios"][0]
    generated_audio_tensor = torch.tensor(generated_audio, requires_grad=True).to(accelerator.device)

    # 생성된 오디오 길이 맞추기(48000=1초)
    if generated_audio_tensor.shape[0] > 480000:
      start = torch.randint(0, generated_audio_tensor.shape[0] - 480000, (1,)).item()
      generated_audio_tensor = generated_audio_tensor[start:start+480000]
    else:
      generated_audio_tensor = torch.nn.functional.pad(generated_audio_tensor, (0, 480000 - generated_audio_tensor.shape[0]))

    # ref_audio 길이 맞추기
    ref_audio = ref_audio[:generated_audio_tensor.shape[0]] 

    # loss 게산 및 학습
    loss = F.mse_loss(generated_audio_tensor, ref_audio)
    accelerator.backward(loss)
    unet_optimizer.step()
    vae_optimizer.step()

    # 유사도 측정
    clap_score = compute_clap_similarity(generated_audio, prompt)

    clap_scores.append(clap_score)
    losses.append(loss.item())

    # wandb 로그
    wandb.log({
        "loss": loss.item(),
        "clap_score": clap_score,
        "audio": wandb.Audio(generated_audio, sample_rate=16000, caption=f"{prompt} step {step+1}")
    })

    step += 1

# LoRA 가중치 저장
save_dir = "./lora_weights"
os.makedirs(save_dir, exist_ok=True)
pipe.unet.save_pretrained(os.path.join(save_dir, "unet_lora"))
pipe.vae.save_pretrained(os.path.join(save_dir, "vae_lora"))

# 평균 기록
wandb.log({
    "average_clap_score": np.mean(clap_scores),
    "average_loss": np.mean(losses)
})
wandb.finish()