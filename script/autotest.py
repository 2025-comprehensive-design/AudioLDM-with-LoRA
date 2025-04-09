import sys
import torch, random, torchaudio, os, json
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. CLAP 기반 유사도 함수

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
        return (similarity + 1) / 2 # [-1,1] -> [0,1] 정규화


# 2. 데이터셋 클래스
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


# 3. 실험 실행 함수
def run_experiment(config):
    wandb.init(project="AudioLDM-with-LoRA", name=config["name"], config=config, group=config["mode"])
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=None,
        log_with="wandb",
        project_config=ProjectConfiguration(project_dir="./", logging_dir="./wandb_logs")
    )
    logger = get_logger(__name__)
    logger.info(accelerator.state, main_process_only=True)

    pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2")
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
    text_encoder.requires_grad_(False)
    pipe.text_encoder = text_encoder.to(accelerator.device)
    pipe.to(accelerator.device)

    optimizer = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, text_encoder.parameters()),
        lr=config["learning_rate"]
    )

    csv_path = "./data/musiccaps/musiccaps-hiphop_real.csv"
    audio_dir = "./data/musiccaps/audio"
    dataset = MusicCapsDataset(csv_path, audio_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    clap_scores, losses = [], []
    audio_index = 1

    for audios, prompts in dataloader:
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
                gen_tensor = F.pad(gen_tensor, (0, ref_audio.shape[0] - gen_tensor.shape[0]))
            loss = F.mse_loss(gen_tensor, ref_audio)
            accelerator.backward(loss)
            optimizer.step()
            clap_score = compute_clap_similarity(gen_audio, prompt)
            wandb.log({
                f"Audio {audio_index} | loss": loss.item(),
                f"Audio {audio_index} | clap_score": clap_score,
                f"Audio {audio_index} | audio": wandb.Audio(gen_audio, sample_rate=16000, caption=prompt)
            })
            clap_scores.append(clap_score)
            losses.append(loss.item())
        audio_index += 1

    os.makedirs("./data/LoRA_weight", exist_ok=True)
    text_encoder.save_pretrained(f"./data/LoRA_weight/{config['name']}")

    wandb.log({
    "average_clap_score": np.mean(clap_scores),
    "average_loss": np.mean(losses)
})
    wandb.summary["final_clap_score"] = float(np.mean(clap_scores))
    wandb.summary["final_loss"] = float(np.mean(losses))
    wandb.finish()

    os.makedirs("./results", exist_ok=True)
    with open(f"./results/result_{config['name']}.json", "w") as f:
        json.dump({
            "experiment": config["name"],
            "targets": config["target_modules"],
            "rank": config["rank"],
            "alpha": config["alpha"],
            "clap_score": float(np.mean(clap_scores)),
            "loss": float(np.mean(losses))
        }, f, indent=4)



# 4. 반복 실험 
if __name__ == "__main__":
    torch.manual_seed(42)
    ranks = [1, 2, 4, 8]
    experiment_sets = {
        "Wq": [f"text_model.encoder.layer.{i}.attention.self.query" for i in range(12)],
        "Wv": [f"text_model.encoder.layer.{i}.attention.self.value" for i in range(12)],
        "WqWv": [f"text_model.encoder.layer.{i}.attention.self.query" for i in range(12)] +
                [f"text_model.encoder.layer.{i}.attention.self.value" for i in range(12)],
    }

    for rank in ranks:
        for weight, target_modules in experiment_sets.items():
            name = f"LoRA_{weight}_rank{rank}"
            config = {
                "mode": "text_encoder",
                "target_modules": target_modules,
                "rank": rank,
                "alpha": 8,
                "learning_rate": 1e-4,
                "name": name,
                "epochs_per_audio": 5
            }
            run_experiment(config)
