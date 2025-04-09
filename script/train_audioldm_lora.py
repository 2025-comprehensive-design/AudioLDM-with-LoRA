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
from torch.utils.data import Dataset, DataLoader, Subset
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
        for attn in ["attn1", "attn2"]
        for proj in ["to_q", "to_v", "to_k"]
    ] + [
        f"up_blocks.{i}.attentions.{j}.transformer_blocks.0.{attn}.{proj}"
        for i in range(3)
        for j in range(3)
        for attn in ["attn1", "attn2"]
        for proj in ["to_q", "to_v", "to_k"]
    ] + [
        f"mid_block.attentions.0.transformer_blocks.0.{attn}.{proj}"
        for attn in ["attn1", "attn2"]
        for proj in ["to_q", "to_v", "to_k"]
    ],
    # vae 타켓 레이어
    "vae_target_modules": [
        f"{i}.mid_block.attentions.0.{proj}"
        for i in ["encoder", "decoder"]
        for proj in ["to_q", "to_v", "to_k"]
    ],
    "rank": 4,  # LoRA 랭크
    "alpha": 8,  # LoRA 스케일링 계수
    "learning_rate": 1e-4,
    "epochs": 3, # 반복 횟수
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
#def compute_clap_similarity(audio_waveform: np.ndarray, text: str) -> float:
#    inputs = processor(audios=audio_waveform, return_tensors="pt", sampling_rate=48000)
#    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
#    with torch.no_grad():
#        audio_embed = clap_model.get_audio_features(**inputs)
#        text_embed = clap_model.get_text_features(**processor(text=text, return_tensors="pt", padding=True).to(accelerator.device))
#        audio_embed = F.normalize(audio_embed, dim=-1)
#        text_embed = F.normalize(text_embed, dim=-1)
#        similarity = (audio_embed @ text_embed.T).item()
#        # [-1, 1] -> [0, 1]로 변환
#        similarity = (similarity + 1) / 2
#    return similarity

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

noise_scheduler = pipe.scheduler
tokenizer = pipe.tokenizer
unet = pipe.unet
text_encoder = pipe.text_encoder
vae = pipe.vae

unet.requires_grad_(True)
vae.requires_grad_(True)
text_encoder.requires_grad_(False)

# unet LoRA 설정
unet_lora_config = LoraConfig(
    r=config["rank"],
    lora_alpha=config["alpha"],
    init_lora_weights="gaussian",
    target_modules=config["unet_target_modules"],
)

unet = get_peft_model(unet, unet_lora_config)

# vae LoRA 설정
vae_lora_config = LoraConfig(
    r=config["rank"],
    lora_alpha=config["alpha"],
    init_lora_weights="gaussian",
    target_modules=config["vae_target_modules"],
)

vae = get_peft_model(vae, vae_lora_config)

unet.to(accelerator.device, dtype=torch.float32)
vae.to(accelerator.device, dtype=torch.float32)
text_encoder.to(accelerator.device, dtype=torch.float32)

# LoRA 파라미터 수 출력
def print_trainable_parameters(model, name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name} 전체 파라미터 수: {total_params:,}")
    print(f"{name} 학습 가능한 LoRA 파라미터 수: {trainable_params:,}")
    print(f"비율: {100 * trainable_params / total_params:.4f}%\n")

pipe.unet = unet
pipe.vae = vae

print_trainable_parameters(unet, name="UNet")
print_trainable_parameters(vae, name="VAE")

# 옵티마이저 설정
unet_optimizer = torch.optim.AdamW(
    params=filter(lambda p: p.requires_grad, unet.parameters()),
    lr=config["learning_rate"]
)
vae_optimizer = torch.optim.AdamW(
    params=filter(lambda p: p.requires_grad, vae.parameters()),
    lr=config["learning_rate"]
)

# 데이터셋 샘플링(musiccaps를 로컬에 다운로드 및 해당 오디오를 다운로드하여 사용함)
class MusicCapsDataset(Dataset):
    def __init__(self, csv_path, audio_dir, target_duration=10, sampling_rate=48000):
        self.audio_dir = audio_dir              # 오디오 파일 경로
        self.target_duration = target_duration  # 10초(오디오 길이)
        self.sampling_rate = sampling_rate      # 오디오 샘플링 레이트(초당 쪼개는 정도)
        self.target_length = self.target_duration * self.sampling_rate  # 오디오 샘플 수

        available_audio_files = set([
            os.path.splitext(fname)[0] 
            for fname in os.listdir(audio_dir)
            if fname.endswith('.wav')
        ])

        # CSV 읽기
        df = pd.read_csv(csv_path)

        # audio_dir에 파일이 있는 것만 필터링
        self.data = df[df["ytid"].isin(available_audio_files)].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # idx에 해당하는 row 가져옴
        row = self.data.iloc[idx]
        # 유튜브 id, 캡션, 시작 시간 가져옴
        ytid = row["ytid"]
        caption = row["caption"]

        audio_path = os.path.join(self.audio_dir, f"{ytid}.wav")
        # 오디오 파일 로드
        audio, sr = torchaudio.load(audio_path)
        audio = audio[0]  # mono 채널만

        # 샘플링 레이트가 다르면 리샘플링
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            audio = resampler(audio)

        if audio.shape[0] > self.target_length:
            audio = audio[:self.target_length]
        elif audio.shape[0] < self.target_length:
            pad_len = self.target_length - audio.shape[0]
            audio = F.pad(audio, (0, pad_len))

        return audio, caption

# 데이터셋 경로
csv_path = "./musiccaps/musiccaps-public.csv"
audio_dir = "./musiccaps/audio"

dataset = MusicCapsDataset(csv_path, audio_dir)
dataset = Subset(dataset, range(0, 100))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 학습
cos_sims = []
losses = []
snrs = []
step = 0

for epoch in range(config["epochs"]):
    for audios, texts in dataloader:
        try:
            unet_optimizer.zero_grad()
            vae_optimizer.zero_grad()
            unet.train()
            vae.train()

            # 오디오와 텍스트 가져오기
            prompt = texts[0]
            ref_audio = audios[0].to(accelerator.device)

            # 오디오 생성(데이터셋의 caption을 사용)
            result = pipe(prompt, num_inference_steps=30, audio_length_in_s=10.0)

            # 오디오 데이터 가져와서 변환
            generated_audio = result["audios"][0]
            generated_audio_tensor = torch.tensor(generated_audio, requires_grad=True).to(accelerator.device)

            # 생성된 오디오 길이 맞추기(48000=1초)
            if generated_audio_tensor.shape[0] > ref_audio.shape[0]:
                generated_audio_tensor = generated_audio_tensor[:ref_audio.shape[0]]
            else:
                pad_len = ref_audio.shape[0] - generated_audio_tensor.shape[0]
                generated_audio_tensor = F.pad(generated_audio_tensor, (0, pad_len))

            # loss 계산 및 학습
            loss = F.mse_loss(generated_audio_tensor, ref_audio)
            accelerator.backward(loss)
            unet_optimizer.step()
            vae_optimizer.step()

            # 유사도 측정
            cos_sim = F.cosine_similarity(
                generated_audio_tensor.unsqueeze(0), 
                ref_audio.unsqueeze(0)
            ).item()
            cos_sim_normalized = (cos_sim + 1) / 2
            
            # SNR 계산
            signal_power = torch.sum(ref_audio ** 2)
            noise_power = torch.sum((ref_audio - generated_audio_tensor) ** 2)
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
            snr = snr.item()

            cos_sims.append(cos_sim_normalized)
            losses.append(loss.item())
            snrs.append(snr)

            # wandb 로그
            wandb.log({
                "loss": loss.item(),
                "cos_sim": cos_sim_normalized,
                "snr_audio": snr,
                "audio": wandb.Audio(generated_audio, sample_rate=16000, caption=f"{prompt} step {step+1}")
            })

            step += 1

        except Exception as e:
            print(f"오류 발생: {e}. 현재 파일 스킵하고 계속 진행합니다.")
            continue

    print(f"Epoch {epoch+1}/{config['epochs']} 완료")

# LoRA 가중치 저장
save_dir = "./lora_weights"
os.makedirs(save_dir, exist_ok=True)
pipe.unet.save_pretrained(os.path.join(save_dir, "unet_lora"))
pipe.vae.save_pretrained(os.path.join(save_dir, "vae_lora"))

# 평균 기록
wandb.log({
    "average_cos_sim": np.mean(cos_sims),
    "average_loss": np.mean(losses),
    "average_snr": np.mean(snrs),
})
wandb.finish()
