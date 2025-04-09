import torch, torchaudio
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from diffusers import AudioLDMPipeline
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import wandb
from transformers import AutoProcessor, ClapModel

# 1. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. CLAP 모델 및 프로세서 로드
processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)

def compute_clap_similarity(audio_waveform: np.ndarray, text: str) -> float:
    inputs = processor(audios=audio_waveform, return_tensors="pt", sampling_rate=48000)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        audio_embed = clap_model.get_audio_features(**inputs)
        text_embed = clap_model.get_text_features(**processor(text=text, return_tensors="pt", padding=True).to(device))
        audio_embed = F.normalize(audio_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        similarity = (audio_embed @ text_embed.T).item()
        return (similarity + 1) / 2

# 3. 데이터셋 클래스 정의
class MusicCapsDataset(Dataset):
    def __init__(self, dataset, target_duration=4, sampling_rate=48000):
        self.dataset = dataset
        self.target_duration = target_duration
        self.sampling_rate = sampling_rate
        self.target_length = target_duration * sampling_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        waveform = torch.tensor(sample["audio"]["array"])
        sr = sample["audio"]["sampling_rate"]
        prompt = sample["caption"]

        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        if waveform.shape[0] > self.target_length:
            waveform = waveform[:self.target_length]
        else:
            waveform = F.pad(waveform, (0, self.target_length - waveform.shape[0]))

        return waveform, prompt

# 4. 설정값
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
    "epochs_per_audio": 5
}

# 5. wandb
wandb.init(project="AudioLDM-with-LoRA", name=config["name"], config=config)

# 6. accelerate 설정
accelerator_project_config = ProjectConfiguration(project_dir="./", logging_dir="AudioLDM-with-LoRA/log")
accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision=None,
    log_with="wandb",
    project_config=accelerator_project_config,
)
logger = get_logger(__name__)
logger.info(accelerator.state, main_process_only=True)

# 7. 모델 준비
pipe = AudioLDMPipeline.from_pretrained(base_model_id)
text_encoder = pipe.text_encoder
lora_config = LoraConfig(
    r=config["rank"],
    lora_alpha=config["alpha"],
    inference_mode=False,
    init_lora_weights="gaussian",
    target_modules=config["target_modules"]
)
text_encoder = get_peft_model(text_encoder, lora_config)
pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(False)
text_encoder.requires_grad_(True)
pipe.text_encoder = text_encoder.to(accelerator.device, dtype=torch.float32)
pipe.to(accelerator.device)

optimizer = torch.optim.AdamW(
    params=filter(lambda p: p.requires_grad, text_encoder.parameters()),
    lr=config["learning_rate"]
)

# 8. 데이터 준비 및 필터링
raw_dataset = load_dataset("mb23/music_caps_4sec_wave_type_classical", split="train")
dataset = MusicCapsDataset(raw_dataset)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 9. 학습 루프
clap_scores, losses = [], []
audio_index = 1

for audios, prompts in dataloader:
    audio_clap_scores, audio_losses = [], []
    prompt = prompts[0]
    ref_audio = audios[0].float().to(accelerator.device)

    for epoch in range(config["epochs_per_audio"]):
        optimizer.zero_grad()

        result = pipe(prompt, num_inference_steps=30, audio_length_in_s=4.0)
        gen_audio = result["audios"][0]
        gen_tensor = torch.tensor(gen_audio, dtype=torch.float32, requires_grad=True).to(accelerator.device)

        if gen_tensor.shape[0] > ref_audio.shape[0]:
            gen_tensor = gen_tensor[:ref_audio.shape[0]]
        else:
            pad_len = ref_audio.shape[0] - gen_tensor.shape[0]
            gen_tensor = F.pad(gen_tensor, (0, pad_len))

        loss = F.mse_loss(gen_tensor, ref_audio)
        accelerator.backward(loss)
        optimizer.step()


        clap_score = compute_clap_similarity(gen_audio, prompt)
        audio_clap_scores.append(clap_score)
        audio_losses.append(loss.item())

        wandb.log({
            f"Audio {audio_index} | loss": loss.item(),
            f"Audio {audio_index} | clap_score": clap_score,
            f"Audio {audio_index} | audio": wandb.Audio(gen_audio, sample_rate=16000, caption=f"Audio {audio_index} Epoch {epoch+1}: {prompt}")
        })

    wandb.log({
        f"Audio {audio_index} | avg_loss": np.mean(audio_losses),
        f"Audio {audio_index} | avg_clap_score": np.mean(audio_clap_scores)
    })

    clap_scores.extend(audio_clap_scores)
    losses.extend(audio_losses)
    audio_index += 1

# 10. LoRA 가중치 저장
save_path = "./data/LoRA_weight/" + config["name"]
text_encoder.save_pretrained(save_path)

wandb.log({
    "average_clap_score": np.mean(clap_scores),
    "average_loss": np.mean(losses)
})
wandb.finish()