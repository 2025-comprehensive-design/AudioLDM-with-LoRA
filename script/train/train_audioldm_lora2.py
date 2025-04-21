

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append("AudioLDM-with-LoRA")

from peft import LoraConfig, get_peft_model, TaskType

from diffusers import AudioLDMPipeline, StableDiffusionPipeline
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

from transformers import ClapTextModelWithProjection, RobertaTokenizerFast, SpeechT5HifiGan
from transformers import TrainingArguments
from transformers import Trainer

import torch

from script.data.datasets import HfAudioDataset
from datasets import load_dataset

base_model_id = "cvssp/audioldm"

base_model = AudioLDMPipeline.from_pretrained(base_model_id)
unet_lora_config = LoraConfig(
        r=2,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_v"],
    )

model = get_peft_model(base_model, unet_lora_config)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=50,
    report_to="none"
)

num_workers = 4
train_batch_size = 2
total_batch_size = train_batch_size * accelerator.num_processes
num_train_epochs = 100
gradient_accumulation_steps = 1
max_train_steps = 1000000
checkpointing_steps = 50000
total_train_loss = 0.0
total_steps = 0

def collate_fn(examples):
    log_mel_spec = torch.stack([example["log_mel_spec"].unsqueeze(0) for example in examples])
    input_ids = torch.stack([example["text"] for example in examples])
    attention_mask = torch.stack([example["attention_mask"] for example in examples])

    return {"log_mel_spec": log_mel_spec, "input_ids": input_ids, "attention_mask" : attention_mask}

dataset = load_dataset("mb23/music_caps_4sec_wave_type_classical", split="train")

# caption에 "guitar" 들어간 것만 필터링
filtered_dataset = dataset.filter(
    lambda example: "guitar" in example["caption"].lower()
)
train_dataset = HfAudioDataset(filtered_dataset)

# 필터링된 데이터셋으로 학습!
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    collate_fn=collate_fn,
    shuffle=True,
    batch_size=train_batch_size,
    num_workers=num_workers,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader,
)

trainer.train()