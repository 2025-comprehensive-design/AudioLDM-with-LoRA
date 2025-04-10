'''
Copyright Dongguk U.V. CSE 2020112030 KIM SEON PYO [2025/03/04 ~]

해당 코드는 DiffusersPipeline의 AudioLDM 모델을 파인튜닝하는 LoRA 가중치를 학습하는 코드입니다.
README.md 를 참고해 주세요

This code is training LoRA weight for AudioLDM in DiffusersPipeline.
you can choose Base_model for training LoRA weight and save at [AudioLDM-with-LoRA/data/LoRA_weight/~]

adapt at app.py to use
'''

# TODO : emd_mask 처리 과정 다시 확인,
# TODO : push_to_hub() 다른 파일에서 처리
# TODO : LoRA weight 처리
# TODO : config.yaml 설정 파일로 바꾸기.
# TODO : noise_offset 확인/ AudioLDM 논문 확인 및 적용
# TODO : LoRA 적용 부분 확인
# TODO : app.py 구축

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append("AudioLDM-with-LoRA")

# import argparse
# import math
import logging
import numpy as np
from contextlib import nullcontext
# from pathlib import Path

import torch, random
import torch.nn.functional as F
import torch.nn as nn

from torchaudio import transforms as AT
from torchvision import transforms as IT

# from torch.utils.data import Dataset
from script.data.datasets import HfAudioDataset
from datasets import load_dataset

# LoRA
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict

from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from diffusers import AudioLDMPipeline
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.optimization import get_scheduler
from transformers import ClapTextModelWithProjection, RobertaTokenizerFast

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

logger = get_logger(__name__)

# 시각화 툴

base_model_id = "cvssp/audioldm-s-full-v2"
dataset_hub_id = ""
validation_prompt = "guitar in hip hop music"
validation_epochs = 1
if is_wandb_available():
    import wandb

num_validation_images = 2

def log_validation(
    pipeline,
    accelerator,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {num_validation_images} mel images with prompt:"
        f" {validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        for _ in range(num_validation_images):
            audio_output = pipeline(validation_prompt, num_inference_steps=30, generator=generator)
            images.append(audio_output.audios[0])

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")

        if tracker.name == "wandb":

            audio_logs = []
            for i, image in enumerate(images):
                audio_logs.append(wandb.Audio(image, sample_rate=16000, caption=f"{i}: {validation_prompt}"))
            tracker.log({phase_name: audio_logs})

            # tracker.log(
            #     {
            #         "validation": [
            #             wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
            #         ]
            #     }
            # )

    return images



def main() :

    accelerator_project_config = ProjectConfiguration(project_dir="./", logging_dir="AudioLDM-with-LoRA/log")

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=None,
        log_with="wandb",
        project_config=accelerator_project_config,
    )
    accelerator.init_trackers(
        project_name="AudioLDM-with-LoRA",
        config=accelerator_project_config,
        init_kwargs={
            "wandb": {
                "entity": "kimsp0317-dongguk-university",
                "group": "gpu-exp-group-1",  # 여러 GPU 실험 묶고 싶을 때
                "tags": ["lora", "audioldm", "guitar"],
                "name": "run-name-optional"
            }
        }
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # model 불러오기
    # pipe = AudioLDMPipeline.from_pretrained(base_model_id)

    unet_config = UNet2DConditionModel.load_config("cvssp/audioldm-s-full-v2", subfolder="unet")

    # 2. 수정 (class conditioning 관련 설정 제거)
    unet_config["class_embed_type"] = None
    unet_config["class_embeddings_concat"] = False
    unet_config["num_class_embeds"] = None  # 또는 0

    if "projection_class_embeddings_input_dim" in unet_config:
        del unet_config["projection_class_embeddings_input_dim"]

    noise_scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")

    tokenizer = RobertaTokenizerFast.from_pretrained(base_model_id, subfolder="tokenizer")

    unet = UNet2DConditionModel.from_config(unet_config)

    text_encoder = ClapTextModelWithProjection.from_pretrained(base_model_id, subfolder="text_encoder")

    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")

    # Class conditioning 관련 레이어 제거 (config에서 제거했으므로 불필요할 수 있음)
    # if hasattr(unet, "class_embedding"):
    #     delattr(unet, "class_embedding")
    # if hasattr(unet, "projection_class_embeddings"):
    #     delattr(unet, "projection_class_embeddings")
    # num_attention_heads = unet.config.num_attention_heads
    attention_head_dim = unet.config.attention_head_dim
    if attention_head_dim is None:
        attention_head_dim = getattr(unet.config, "projection_dim", None)

    if attention_head_dim is None:
        raise ValueError("UNet config does not have 'attention_head_dim' or 'projection_dim'.")

    # 기존 모델의 가중치는 잠금
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_v", "to_out.0"],
    )
    # medel = get_peft_model(unet, unet_lora_config)

    unet.add_adapter(unet_lora_config)

    weight_dtype = torch.float32
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        lora_layers,
        lr=1.0e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-08,
    )

    num_workers = 4
    train_batch_size = 1
    total_batch_size = train_batch_size * accelerator.num_processes
    num_train_epochs = 1
    gradient_accumulation_steps = 1
    max_train_steps = 1000000
    checkpointing_steps = 50000
    total_train_loss = 0.0
    total_steps = 0

    def collate_fn(examples):
        log_mel_spec = torch.stack([example["log_mel_spec"].unsqueeze(0) for example in examples])
        input_ids = torch.stack([example["text"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples]) # attention mask 스택
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
    output_dir = "/AudioLDM-with-LoRA/data/LoRA_weight"
    
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

    noise_offset = 0.0015 # in AudioLDM

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
        epoch_total_loss = 0.0
        num_steps_per_epoch = 0

        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}",
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in progress_bar:
            num_steps_per_epoch += 1
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["log_mel_spec"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)

                # if noise_offset:
                #     noise += noise_offset * torch.randn(
                #         (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                #     )

                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_outputs = text_encoder(
                    batch["input_ids"], attention_mask=batch["attention_mask"], return_dict=True
                )
                attention_mask = encoder_outputs.get("attention_mask")

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=attention_mask,
                    attention_mask=attention_mask,
                    return_dict=False
                )[0]

                target = noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                # avg_loss = accelerator.gather(loss).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps
                epoch_total_loss += avg_loss.item()
                total_train_loss += avg_loss.item()
                total_steps += 1

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    progress_bar.set_postfix({"loss": train_loss})
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0
                    global_step += 1

                    if global_step % checkpointing_steps == 0 and accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        # unwrapped_unet = unwrap_model(unet)
                        # unet.save_lora_weights(save_directory=save_path, unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet)), safe_serialization=True)
                        # logger.info(f"Saved state to {save_path}")

            if global_step >= max_train_steps:
                break

            accelerator.log({"total_train_loss": total_train_loss / total_steps if total_steps > 0 else 0.0}, step=global_step)

        if accelerator.is_main_process and validation_prompt is not None and epoch % validation_epochs == 0:
            pipeline = AudioLDMPipeline.from_pretrained(base_model_id, unet=unwrap_model(unet), torch_dtype=weight_dtype)
            images = log_validation(pipeline, accelerator, epoch)
            del pipeline
            torch.cuda.empty_cache()

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        AudioLDMPipeline.push_to_hub("Rofla/AudioLDM-with-LoRA", unet_lora_state_dict)

        if validation_prompt is not None:
            pipeline = AudioLDMPipeline.from_pretrained(base_model_id, torch_dtype=weight_dtype)
            images = log_validation(pipeline, accelerator, epoch, is_final_validation=True)

    accelerator.end_training()

if __name__ == "__main__":
    main()