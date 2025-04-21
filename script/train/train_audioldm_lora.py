'''
Copyright Dongguk U.V. CSE 2020112030 KIM SEON PYO [2025/03/04 ~]

해당 코드는 DiffusersPipeline의 AudioLDM 모델을 파인튜닝하는 LoRA 가중치를 학습하는 코드입니다.
README.md 를 참고해 주세요

This code is training LoRA weight for AudioLDM in DiffusersPipeline.
you can choose Base_model for training LoRA weight and save at [AudioLDM-with-LoRA/data/LoRA_weight/~]

adapt at app.py to use
'''

# TODO : push_to_hub() 다른 파일에서 처리
# TODO : LoRA weight 처리
# TODO : config.yaml 설정 파일로 바꾸기.
# TODO : LoRA 적용 부분 확인
# TODO : app.py 구축
# TODO : LoRA 학습 처리 확인

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
from diffusers.utils import check_min_version, is_wandb_available # , convert_state_dict_to_diffusers
# from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.optimization import get_scheduler
from transformers import ClapTextModelWithProjection, RobertaTokenizerFast, SpeechT5HifiGan

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

import librosa
import librosa.display
import io
from PIL import Image

logger = get_logger(__name__)

# 시각화 툴

base_model_id = "cvssp/audioldm-s-full-v2"
dataset_hub_id = ""
validation_prompt = "guitar in hip hop music"
validation_epochs = 1
if is_wandb_available():
    import wandb

num_validation_images = 2

def plot_spectrogram_to_image(spec, title=None):
    """Spectrogram NumPy 배열을 PIL 이미지로 변환합니다."""
    plt.figure(figsize=(10, 4))
    # dB 스케일로 변환된 spectrogram을 사용한다고 가정
    img = librosa.display.specshow(spec, sr=16000, hop_length=512,
                                   x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(img, format='%+2.0f dB')
    if title:
        plt.title(title)
    plt.tight_layout()

    # Plot을 BytesIO 버퍼에 PNG 형식으로 저장
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close() # Matplotlib figure 메모리 해제
    buf.seek(0)
    # 버퍼에서 PIL 이미지 로드
    image = Image.open(buf)
    return image

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
    mel_spectrogram_images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        for i in range(num_validation_images):
            audio_output = pipeline(validation_prompt, num_inference_steps=30, generator=generator)
            images.append(audio_output.audios[0])
            
                # librosa를 사용하여 Mel Spectrogram 계산
            mel_spec = librosa.feature.melspectrogram(
                    y=audio_output.audios[0],
                    sr=16000,
                    n_fft=1024,
                    hop_length=160,
                    n_mels=64
                )
                # Power spectrogram을 dB 스케일로 변환
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                # Spectrogram 배열을 시각화된 PIL 이미지로 변환
            spec_image = plot_spectrogram_to_image(mel_spec_db, title=f"Spectrogram {i}: {validation_prompt}")
            mel_spectrogram_images.append(spec_image)

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

            spec_image_logs = []
            for i, spec_img in enumerate(mel_spectrogram_images):
                spec_image_logs.append(wandb.Image(spec_img, caption=f"{phase_name} Spectrogram {i}: {validation_prompt}"))
            tracker.log({f"{phase_name}_spectrogram": spec_image_logs})

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
                "group": "gpu-exp-group-1",
                "tags": ["lora", "audioldm", "guitar"],
                "name": "<task : r = 2, alpha = 4>"
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

    # unet_config = UNet2DConditionModel.load_config(base_model_id, subfolder="unet")
    # unet_config["num_class_embeds"] = None  # 또는 0

    ### model load
    # unet = UNet2DConditionModel.from_config(unet_config)
    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    # pipe = AudioLDMPipeline.from_pretrained(base_model_id, unet=unet)

    noise_scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    tokenizer = RobertaTokenizerFast.from_pretrained(base_model_id, subfolder="tokenizer")
    text_encoder = ClapTextModelWithProjection.from_pretrained(base_model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    vocoder = SpeechT5HifiGan.from_pretrained(base_model_id, subfolder="vocoder")

    # 기존 모델의 가중치는 잠금
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=2,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_v"],
    )

    unet = get_peft_model(unet, unet_lora_config)
    # unet.add_adapter(unet_lora_config, "default")
    # unet.set_adapter("default")

    weight_dtype = torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        lora_layers,
        lr=1.0e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
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

    lr_scheduler = get_scheduler(
        "polynomial",
        optimizer=optimizer,
        num_warmup_steps = 0, #500 * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    ### Train
    output_dir = "./AudioLDM-with-LoRA/data/LoRA_weight"
    
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

    # noise_offset = 0.0015 # in AudioLDM

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        optimizer.zero_grad()

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
                latents = vae.encode(batch["log_mel_spec"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                
                noise = torch.randn_like(latents)

                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                input_ids = batch["input_ids"].to(latents.device)   
                attention_mask = batch["attention_mask"].to(latents.device)
                
                # 3차원 텐서를 2차원으로 변환
                input_ids = input_ids.squeeze(1)
                attention_mask = attention_mask.squeeze(1)

                # 토큰 길이 제한 확인
                untruncated_ids = tokenizer(
                    tokenizer.batch_decode(input_ids),
                    padding="longest",
                    return_tensors="pt"
                ).input_ids.to(latents.device)

                if untruncated_ids.shape[-1] >= input_ids.shape[-1] and not torch.equal(
                    input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(
                        untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                    )
                    

                encoder_outputs = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True
                )

                # text_embeds 직접 사용
                prompt_embeds = encoder_outputs.text_embeds
                
                # L2 normalization 적용
                prompt_embeds = F.normalize(prompt_embeds, dim=-1)
                
                # 배치 차원에 대해 반복하여 확장
                bs_embed, seq_len = prompt_embeds.shape
                num_waveforms_per_prompt = 1  # 학습 시에는 1로 설정
                prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt)
                prompt_embeds = prompt_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len)

                # negative prompt 처리 (classifier-free guidance)
                uncond_tokens = [""] * bsz
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=input_ids.shape[1],
                    truncation=True,
                    return_tensors="pt",
                )
                uncond_input_ids = uncond_input.input_ids.to(latents.device)
                uncond_attention_mask = uncond_input.attention_mask.to(latents.device)

                negative_prompt_embeds = text_encoder(
                    uncond_input_ids,
                    attention_mask=uncond_attention_mask,
                    return_dict=True,
                ).text_embeds
                negative_prompt_embeds = F.normalize(negative_prompt_embeds, dim=-1)
                
                # negative prompt도 동일하게 확장
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_waveforms_per_prompt)
                negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len)

                # unconditional과 conditional embeddings를 concatenate
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

                # latents를 classifier-free guidance를 위해 확장
                latent_model_input = torch.cat([noisy_latents] * 2)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timesteps)

                # timesteps도 동일하게 확장
                timesteps = timesteps.repeat(2)

                # attention mask도 동일하게 확장
                attention_mask = torch.cat([attention_mask] * 2)

                target = noise

                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=None,
                    # encoder_attention_mask=attention_mask,
                    class_labels=prompt_embeds,
                    return_dict=False
                )[0]

                # unconditional과 conditional prediction 분리
                noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
                
                # guidance scale 적용 (학습 시에는 1.0으로 설정)
                guidance_scale = 1.0
                model_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps
                epoch_total_loss += avg_loss.item()
                total_train_loss += avg_loss.item()
                total_steps += 1

                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": train_loss})
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                global_step += 1

                if global_step % checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    unwrapped_unet = unwrap_model(unet)
                    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
                    # unet.save_pretrained(save_path)
                    # unet.save_lora_weights(save_directory=save_path, unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet)), safe_serialization=True)
                    # logger.info(f"Saved state to {save_path}")

            accelerator.log({"total_train_loss": total_train_loss / total_steps if total_steps > 0 else 0.0}, step=global_step)
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        if accelerator.is_main_process and validation_prompt is not None and epoch % validation_epochs == 0:
            unwrapped_unet = unwrap_model(unet)
            pipeline = AudioLDMPipeline.from_pretrained(base_model_id, unet=unwrapped_unet, torch_dtype=weight_dtype)
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
        # AudioLDMPipeline.push_to_hub("Rofla/AudioLDM-with-LoRA", unet_lora_state_dict)

        if validation_prompt is not None:
            pipeline = AudioLDMPipeline.from_pretrained(base_model_id, unet=unet_lora_state_dict, torch_dtype=weight_dtype)
            images = log_validation(pipeline, accelerator, epoch, is_final_validation=True)

    accelerator.end_training()

if __name__ == "__main__":
    main()