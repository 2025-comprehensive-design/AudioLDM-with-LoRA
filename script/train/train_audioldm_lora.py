'''
Copyright Dongguk U.V. CSE 2020112030 KIM SEON PYO [2025/03/04 ~]

해당 코드는 DiffusersPipeline의 AudioLDM 모델을 파인튜닝하는 LoRA 가중치를 학습하는 코드입니다.
README.md 를 참고해 주세요

This code is training LoRA weight for AudioLDM in DiffusersPipeline.
you can choose Base_model for training LoRA weight and save at [AudioLDM-with-LoRA/data/LoRA_weight/~]

adapt at app.py to use 
'''
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append("AudioLDM-with-LoRA")

import argparse
import logging
import math
import numpy as np
from pathlib import Path

import torch, random, torchaudio
import torch.nn.functional as F
import torch.nn as nn
from torchaudio import transforms as AT
from torchvision import transforms as IT
from torchvision import transforms

from torch.utils.data import Dataset
import datasets
from script.data.datasets import AudioDataset, HfAudioDataset
from datasets import load_dataset

# LoRA 
from peft import LoraConfig

from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from transformers import RobertaTokenizerFast, ClapTextModelWithProjection

from diffusers import AudioLDMPipeline
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.optimization import get_scheduler

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

logger = get_logger(__name__)

# 시각화 툴

base_model_id = "cvssp/audioldm-s-full-v2"
dataset_hub_id = ""

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

    # model 불러오기
    pipe = AudioLDMPipeline.from_pretrained(base_model_id)
    noise_scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    vae = pipe.vae

    # 기존 모델의 가중치는 잠금
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
    def preprocess_train(example):
        mel = waveform_to_mel_tensor(torch.tensor(example["audio"]["array"]),
                                    sampling_rate=example["audio"]["sampling_rate"])
        
        # tokenizer 결과 전체 반환
        tokenized = tokenize_captions({"caption": [example["caption"]]})  # ❗ dict
        return {
            "audio": mel,
            "caption": {k: v[0] for k, v in tokenized.items()}  # 각 key에서 첫 번째 값만 추출 (1개 샘플이므로)
        }

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples["caption"]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    mel_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def waveform_to_mel_tensor(y, sampling_rate=16000, n_fft=1024, hop_length=256, win_length=1024,
                           n_mels=64, mel_fmin=0.0, mel_fmax=8000.0, device="cpu",
                           mel_basis_cache={}, hann_window_cache={}):
        # numpy → torch
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        if y.ndim == 1:
            y = y.unsqueeze(0)  # [1, T]

        if torch.min(y) < -1.0:
            print("train min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("train max value is ", torch.max(y))

        mel_key = f"{mel_fmax}_{device}"

        # mel basis caching
        if mel_key not in mel_basis_cache:
            import librosa
            mel = librosa.filters.mel(sr=sampling_rate,
                                    n_fft=n_fft,
                                    n_mels=n_mels,
                                    fmin=mel_fmin,
                                    fmax=mel_fmax)
            mel_basis_cache[mel_key] = torch.from_numpy(mel).float().to(device)

        if device not in hann_window_cache:
            hann_window_cache[device] = torch.hann_window(win_length).to(device)

        # padding
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode='reflect'
        ).squeeze(1)

        # STFT
        stft_spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=hann_window_cache[device],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        stft_spec = torch.abs(stft_spec).float()  # magnitude

        mel_spec = torch.matmul(mel_basis_cache[mel_key], stft_spec)
        mel_spec = spectral_normalize_torch(mel_spec)

        return mel_spec[0]  # [n_mels, time]
    
    def spectral_normalize_torch(magnitudes, clip_val=1e-5):
        return torch.log(torch.clamp(magnitudes, min=clip_val))

    # 데이터셋 로드 코드 작성
    def collate_fn(examples):
        # MEL 이미지 변환
        mel_tensors = [waveform_to_mel_tensor(torch.tensor(example["audio"]["array"]),
                                      sampling_rate=example["audio"]["sampling_rate"]) for example in examples]
        mel = torch.stack(mel_tensors)

        # 캡션 tokenizer 적용
        raw_captions = [example["caption"] for example in examples]  # str 목록
        tokenized = tokenize_captions({"caption": raw_captions}) 

        return {
            "audio": mel,
            "caption": tokenized
        }
    
    
    dataset = load_dataset("mb23/music_caps_4sec_wave_type_classical", split="train")
    train_dataset = HfAudioDataset(dataset)

    # # caption에 "hiphop" 들어간 것만 필터링
    # filtered_dataset = dataset.filter(
    #     lambda example: "hiphop" in example["caption"].lower()
    # )

    num_workers = 0
    train_batch_size = 16
    total_batch_size = train_batch_size * accelerator.num_processes * accelerator.num_processes
    num_train_epochs = 100
    gradient_accumulation_steps = 1
    max_train_steps = 15000

    # 필터링된 데이터셋으로 학습!
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_batch_size,
        num_workers=16,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=500 * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    ### Train
    output_dir = "AudioLDM-with-LoRA/data/LoRA_weight"


    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    noise_offset = 8

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                print(f"batch : {batch['caption'].shape}")
                # Convert images to latent space
                latents = vae.encode(batch["audio"].unsqueeze(1).to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if noise_offset:
                    noise += noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                batch_size = noisy_latents.shape[0]

                text_input = tokenizer(
                    "asfd", padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
                )
                with torch.no_grad():
                    text_embeddings = text_encoder(text_input.input_ids.to(latents.device))[0]

                max_length = text_input.input_ids.shape[-1]
                uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
                uncond_embeddings = text_encoder(uncond_input.input_ids.to(latents.device))[0]
                uncond_embeddings = text_encoder(batch["caption"], return_dict=False)[0]
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                encoder_hidden_states = text_embeddings

                target = noise

                class_labels = torch.zeros(batch_size, 512).to(dtype=latents.dtype, device=latents.device)

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    class_labels=class_labels,
                    return_dict=False
                )[0]

                # Predict the noise residual and compute loss
                # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item()

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        unwrapped_unet = unwrap_model(unet)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        AudioLDMPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                # create pipeline
                pipeline = AudioLDMPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                images = log_validation(pipeline, args, accelerator, epoch)

                del pipeline
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        AudioLDMPipeline.save_lora_weights(
            save_directory=output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

        # Final inference
        # Load previous pipeline
        # if args.validation_prompt is not None:
        #     pipeline = AudioLDMPipeline.from_pretrained(
        #         args.pretrained_model_name_or_path,
        #         revision=args.revision,
        #         variant=args.variant,
        #         torch_dtype=weight_dtype,
        #     )

            # load attention processors
            # pipeline.load_lora_weights(output_dir)

            # run inference
            # images = log_validation(pipeline, args, accelerator, epoch, is_final_validation=True)

    accelerator.end_training()



if __name__ == "__main__":
    main()