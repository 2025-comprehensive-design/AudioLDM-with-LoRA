'''
Copyright Dongguk U.V. CSE 2020112030 KIM SEON PYO [2025/03/04 ~]

해당 코드는 DiffusersPipeline의 AudioLDM 모델을 파인튜닝하는 기법 중 하나인 LoRA에 관한 가중치를 학습하는 코드입니다.
README.md 를 참고해 주세요

This code is training LoRA weight for AudioLDM in DiffusersPipeline.
you can choose Base_model for training LoRA weight and save at [AudioLDM-with-LoRA/data/LoRA_weight/~]

adapt at app.py to use 
'''

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, ClapModel
from diffusers import AudioLDMPipeline
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
import wandb

logger = get_logger(__name__)
torch.manual_seed(42)


base_model_id = "cvssp/audioldm-s-full-v2"

# 환경변수에서 동적 로딩
rank = int(os.environ.get("LORA_RANK", 16))
alpha = int(os.environ.get("LORA_ALPHA", 8))
target_modules = os.environ.get("LORA_TARGET_MODULES", "to_q,to_v").split(',')

# 유닛 설정 함수
def generate_lora_targets(modules):
    down = [
        f"down_blocks.{i}.attentions.{j}.transformer_blocks.0.{attn}.{proj}"
        for i in range(1, 4) for j in range(2)
        for attn in ['attn1', 'attn2'] for proj in modules
    ]
    up = [
        f"up_blocks.{i}.attentions.{j}.transformer_blocks.0.{attn}.{proj}"
        for i in range(3) for j in range(3)
        for attn in ['attn1', 'attn2'] for proj in modules
    ]
    mid = [
        f"mid_block.attentions.0.transformer_blocks.0.{attn}.{proj}"
        for attn in ['attn1', 'attn2'] for proj in modules
    ]
    return down + up + mid

# VAE용 타겟: mid_block attention 8개 기기
def generate_vae_targets(modules):
    allowed_proj = {"to_q", "to_k", "to_v", "to_out.0"}
    return [
        f"{block}.mid_block.attentions.0.{proj}"
        for block in ["encoder", "decoder"]
        for proj in modules if proj in allowed_proj
    ]

# 환경변수 처리
vae_custom_targets = os.environ.get("LORA_VAE_TARGETS", "")
if vae_custom_targets.strip():
    vae_target_modules = [v.strip() for v in vae_custom_targets.split(',')]
else:
    vae_target_modules = generate_vae_targets(target_modules)

# config 정의
config = {
    "unet_target_modules": generate_lora_targets(target_modules),
    "vae_target_modules": vae_target_modules,
    "rank": rank,
    "alpha": alpha,
    "learning_rate": 1e-4,
    "max_train_steps": 1000,
}



tags = [f"rank{rank}", f"alpha{alpha}"] + target_modules

wandb.init(project="audioldm-lora-experiment", name=f"rank{rank}_alpha{alpha}_{'-'.join(target_modules)}", config=config, tags=tags)

accelerator_project_config = ProjectConfiguration(project_dir="./", logging_dir="log")
accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision=None,
    log_with="wandb",
    project_config=accelerator_project_config,
)

processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(accelerator.device)

def compute_clap_similarity(audio_tensor: torch.Tensor, text: str) -> float:
    inputs = processor(
        audios=audio_tensor.detach().cpu().numpy(),
        return_tensors="pt",
        sampling_rate=48000
    )
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    with torch.no_grad():
        audio_embed = clap_model.get_audio_features(**inputs)
        text_embed = clap_model.get_text_features(**processor(text=text, return_tensors="pt", padding=True).to(accelerator.device))
        audio_embed = F.normalize(audio_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        similarity = (audio_embed @ text_embed.T).item()
        return (similarity + 1) / 2

if torch.backends.mps.is_available():
    accelerator.native_amp = False

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger.info(accelerator.state, main_process_only=False)

pipe = AudioLDMPipeline.from_pretrained(base_model_id)
pipe.to(accelerator.device)
pipe.unet.requires_grad_(True)
pipe.vae.requires_grad_(True)
pipe.text_encoder.requires_grad_(False)

pipe.unet = get_peft_model(pipe.unet, LoraConfig(r=config["rank"], lora_alpha=config["alpha"], init_lora_weights="gaussian", target_modules=config["unet_target_modules"]))
pipe.vae = get_peft_model(pipe.vae, LoraConfig(r=config["rank"], lora_alpha=config["alpha"], init_lora_weights="gaussian", target_modules=config["vae_target_modules"]))

unet_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, pipe.unet.parameters()), lr=config["learning_rate"])
vae_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, pipe.vae.parameters()), lr=config["learning_rate"])

dataset = load_dataset("deetsadi/musiccaps_spectrograms", split="train")
filtered = dataset.filter(lambda x: any(tag in x["caption"].lower() for tag in ["hiphop", "drill", "rapping"])) #태깅을 최대한 가능하게 rap도 추가했어요...

class SpectrogramDataset(Dataset):
    def __init__(self, dataset, max_samples=3000):
        self.dataset = dataset.select(range(min(len(dataset), max_samples)))
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.transform(item["mel"]).squeeze(0), item["caption"]

dataloader = DataLoader(SpectrogramDataset(filtered, config["max_train_steps"]), batch_size=1, shuffle=True)

clap_scores, losses = [], []
step = 0
for spectrograms, texts in dataloader:
    if step >= config["max_train_steps"]: break

    unet_optimizer.zero_grad(); vae_optimizer.zero_grad()
    prompt, ref_spec = texts[0], spectrograms[0].to(accelerator.device)
    result = pipe(prompt, num_inference_steps=30, audio_length_in_s=10.0)
    audio_tensor = torch.tensor(result["audios"][0], requires_grad=True).to(accelerator.device)

    clap_score = compute_clap_similarity(audio_tensor, prompt) #Contrastive Language–Audio Pretraining Score... 코사인 유사도임 결국.
    loss = 1 - clap_score
    loss_tensor = torch.tensor(loss, requires_grad=True, device=accelerator.device)

    accelerator.backward(loss_tensor)
    unet_optimizer.step(); vae_optimizer.step()

    clap_scores.append(clap_score); losses.append(loss_tensor.item())
    wandb.log({"loss": loss_tensor.item(), "clap_score": clap_score, "average_loss": np.mean(losses), "average_clap_score": np.mean(clap_scores), "audio": wandb.Audio(result["audios"][0], sample_rate=16000, caption=f"{prompt} step {step+1}")})
    step += 1

pipe.unet.save_pretrained("./lora_weights/unet_lora")
pipe.vae.save_pretrained("./lora_weights/vae_lora")
wandb.log({"average_clap_score": np.mean(clap_scores), "average_loss": np.mean(losses)})
wandb.finish()
