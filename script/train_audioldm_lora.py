import sys
import torch, random, torchaudio, os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from diffusers import AudioLDMPipeline
from peft import LoraConfig, get_peft_model
import wandb  
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration  
import torch.nn.functional as F
from transformers import AutoProcessor, ClapModel

# 1. 디바이스 설정 (GPU 사용 가능 시 CUDA 우선)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. CLAP 모델 및 전처리기 로드 (텍스트-오디오 유사도 계산용)
processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)

# 3. CLAP 유사도 계산 함수 정의
def compute_clap_similarity(audio_waveform: np.ndarray, text: str) -> float:
    inputs = processor(audios=audio_waveform, return_tensors="pt", sampling_rate=48000)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        audio_embed = clap_model.get_audio_features(**inputs)
        text_embed = clap_model.get_text_features(**processor(text=text, return_tensors="pt", padding=True).to(device))
        audio_embed = F.normalize(audio_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        similarity = (audio_embed @ text_embed.T).item()
        return (similarity + 1) / 2  # [-1,1] → [0,1] 정규화

# 4. MusicCaps 데이터셋 클래스 정의
class MusicCapsDataset(Dataset):
    def __init__(self, csv_path, audio_dir, max_samples=100, target_duration=10, sampling_rate=48000):
        self.data = pd.read_csv(csv_path).head(max_samples)
        self.audio_dir = audio_dir
        self.target_duration = target_duration
        self.sampling_rate = sampling_rate
        self.target_length = target_duration * sampling_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        ytid = row["ytid"]
        caption = row["caption"]
        start_s = row.get("start_s", 0)

        audio_path = os.path.join(self.audio_dir, f"{ytid}.wav")
        audio, sr = torchaudio.load(audio_path)
        audio = audio[0]

        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            audio = resampler(audio)

        start_sample = int(start_s * self.sampling_rate)
        end_sample = start_sample + self.target_length

        if end_sample > audio.shape[0]:
            audio = torch.nn.functional.pad(audio, (0, end_sample - audio.shape[0]))

        audio = audio[start_sample:end_sample]
        return audio, caption

# 5. 실험 설정값 및 seed 고정
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
    "learning_rate": 1e-4,
    "name": "LoRA_WqWv_rank1",
    "epochs_per_audio": 10
}

# 6. wandb 실험 초기화
run = wandb.init(
    project="AudioLDM-with-LoRA",
    name=config["name"],
    config=config,
)

# 7. accelerate 설정
accelerator_project_config = ProjectConfiguration(project_dir="./", logging_dir="AudioLDM-with-LoRA/log")
accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision=None,
    log_with="wandb",
    project_config=accelerator_project_config,
)

logger = get_logger(__name__)
logger.info(accelerator.state, main_process_only=True)

# 8. AudioLDM 모델 로드
pipe = AudioLDMPipeline.from_pretrained(base_model_id)

# 9. LoRA 적용 (text_encoder에만)
text_encoder = pipe.text_encoder
lora_config = LoraConfig(
    r=config["rank"],
    lora_alpha=config["alpha"],
    inference_mode=False,
    init_lora_weights="gaussian",
    target_modules=config["target_modules"]
)
text_encoder = get_peft_model(text_encoder, lora_config)

# 10. 학습 대상 설정 (text_encoder만 학습)
pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(False)
text_encoder.requires_grad_(True)

# 11. 모델을 accelerator 디바이스에 할당
pipe.to(accelerator.device)
text_encoder.to(accelerator.device, dtype=torch.float32)
pipe.text_encoder = text_encoder

# 12. 옵티마이저 정의
optimizer = torch.optim.AdamW(
    params=filter(lambda p: p.requires_grad, text_encoder.parameters()),
    lr=config["learning_rate"]
)

# 13. 데이터셋 로드 및 DataLoader 설정
csv_path = "./data/musiccaps/musiccaps-hiphop_real.csv"
audio_dir = "./data/musiccaps/audio"
dataset = MusicCapsDataset(csv_path, audio_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 14. 학습 루프 시작
clap_scores = []
losses = []
audio_index = 1

for audios, texts in dataloader:
    audio_clap_scores = []
    audio_losses = []
    prompt = texts[0]
    ref_audio = audios[0].to(accelerator.device)

    for epoch in range(config["epochs_per_audio"]):
        optimizer.zero_grad()

        result = pipe(prompt, num_inference_steps=30, audio_length_in_s=10.0)
        audio = result["audios"][0]
        audio_tensor = torch.tensor(audio, requires_grad=True).to(accelerator.device)

        if audio_tensor.shape[0] > ref_audio.shape[0]:
            audio_tensor = audio_tensor[:ref_audio.shape[0]]
        else:
            pad_len = ref_audio.shape[0] - audio_tensor.shape[0]
            audio_tensor = F.pad(audio_tensor, (0, pad_len))

        loss = F.mse_loss(audio_tensor, ref_audio)
        accelerator.backward(loss)
        optimizer.step()

        clap_score = compute_clap_similarity(audio, prompt)

        audio_clap_scores.append(clap_score)
        audio_losses.append(loss.item())

        wandb.log({
            "step": epoch + 1,
            f"Audio {audio_index} | loss": loss.item(),
            f"Audio {audio_index} | clap_score": clap_score,
            f"Audio {audio_index} | audio": wandb.Audio(audio, sample_rate=16000, caption=f"Audio {audio_index} Epoch {epoch+1}: {prompt}")
        })

    wandb.log({
        f"Audio {audio_index} | avg_loss": np.mean(audio_losses),
        f"Audio {audio_index} | avg_clap_score": np.mean(audio_clap_scores)
    })

    clap_scores.extend(audio_clap_scores)
    losses.extend(audio_losses)
    audio_index += 1

# 15. LoRA 가중치 저장
save_path = "./data/LoRA_weight/" + config["name"]
text_encoder.save_pretrained(save_path)

# 16. 전체 평균 성능 기록
wandb.log({
    "average_clap_score": np.mean(clap_scores),
    "average_loss": np.mean(losses)
})
wandb.finish()
