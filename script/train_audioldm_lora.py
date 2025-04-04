'''
Copyright Dongguk U.V. CSE 2020112030 KIM SEON PYO [2025/03/04 ~]

해당 코드는 DiffusersPipeline의 AudioLDM 모델을 파인튜닝하는 LoRA 가중치를 학습하는 코드입니다.
README.md 를 참고해 주세요

This code is training LoRA weight for AudioLDM in DiffusersPipeline.
you can choose Base_model for training LoRA weight and save at [AudioLDM-with-LoRA/data/LoRA_weight/~]

adapt at app.py to use 
'''

import argparse
import logging
import math
import os
import numpy as np
from pathlib import Path

import torch, random
import torch.nn.functional as F
from torchaudio import transforms as AT
from torchvision import transforms as IT

# LoRA 
from peft import LoraConfig

from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from transformers import RobertaTokenizerFast, ClapTextModelWithProjection

from diffusers import AudioLDMPipeline
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import is_wandb_available

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

logger = get_logger(__name__)

# 시각화 툴

base_model_id = "cvssp/audioldm-s-full-v2"

if is_wandb_available():
    import wandb

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="kimsp0317-dongguk-university",
    # Set the wandb project where this run will be logged.
    project="AudioLDM-with-LoRA",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

def main() :
    accelerator_project_config = ProjectConfiguration(project_dir="./", logging_dir="AudioLDM-with-LoRA/log")
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=None,
        log_with="wandb",
        project_config=accelerator_project_config,
    )
    
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Load scheduler, tokenizer and models.
    scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    tokenizer = RobertaTokenizerFast.from_pretrained(
    base_model_id, subfolder="tokenizer")
    text_encoder = ClapTextModelWithProjection.from_pretrained(
        base_model_id, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        base_model_id, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        base_model_id, subfolder="unet"
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet_lora_config = LoraConfig(
    r=2,
    lora_alpha=2,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    
    unet.add_adapter(unet_lora_config)

    weight_dtype = torch.float32
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        lora_layers,
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-08,
    )
    ###
    # 데이터셋 로드 코드 작성
    
    ###




if __name__ == "__main__":
    main()