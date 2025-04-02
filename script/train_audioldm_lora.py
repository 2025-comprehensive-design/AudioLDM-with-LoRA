'''
Copyright Dongguk U.V. CSE 2020112030 KIM SEON PYO [2025/03/04 ~]

해당 코드는 DiffusersPipeline의 AudioLDM 모델을 파인튜닝하는 기법 중 하나인 LoRA에 관한 가중치를 학습하는 코드입니다.
README.md 를 참고해 주세요

This code is training LoRA weight for AudioLDM in DiffusersPipeline.
you can choose Base_model for training LoRA weight and save at [AudioLDM-with-LoRA/data/LoRA_weight/~]

adapt at app.py to use 
'''
import torch, random
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from diffusers import AudioLDMPipeline
from torchaudio import transforms as AT
from torchvision import transforms as IT

# 시각화 툴
import wandb

base_model_id = "cvssp/audioldm-s-full-v2"

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="my-awesome-team-name",
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
