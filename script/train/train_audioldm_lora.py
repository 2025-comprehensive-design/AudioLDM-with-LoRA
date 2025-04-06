'''
Copyright Dongguk U.V. CSE 2020112030 KIM SEON PYO [2025/03/04 ~]

해당 코드는 DiffusersPipeline의 AudioLDM 모델을 파인튜닝하는 LoRA 가중치를 학습하는 코드입니다.
README.md 를 참고해 주세요

This code is training LoRA weight for AudioLDM in DiffusersPipeline.
you can choose Base_model for training LoRA weight and save at [AudioLDM-with-LoRA/data/LoRA_weight/~]

adapt at app.py to use 
'''
import sys

sys.path.append("AudioLDM-with-LoRA")

import argparse
import logging
import math
import os
import numpy as np
from pathlib import Path

import torch, random, torchaudio
import torch.nn.functional as F
from torchaudio import transforms as AT
from torchvision import transforms as IT
from torchvision import transforms

from torch.utils.data import Dataset
import datasets

# LoRA 
from peft import LoraConfig

from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from transformers import RobertaTokenizerFast, ClapTextModelWithProjection

from diffusers import AudioLDMPipeline
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

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
def load_dataset(dataset_name=None, cache_dir=None, train_data_dir=None):
    if dataset_name is not None:
        dataset = datasets.load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            data_dir=train_data_dir,
        )
    else:
        data_files = {}
        if train_data_dir is not None:
            data_files["train"] = os.path.join(train_data_dir, "**")
        dataset = datasets.load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=cache_dir,
        )
    return dataset

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

    # 기존 UNet의 config를 복사
    unet_config = pipe.unet.config
    unet_config.num_class_embeds = 0
    unet_config.class_embed_type = None

    # 수정된 config로 UNet을 새로 생성
    unet = UNet2DConditionModel.from_config(unet_config)

    # 기존 pretrained UNet의 weight 중 일치하는 부분만 로딩
    pretrained_unet = UNet2DConditionModel.from_pretrained(
        base_model_id,
        subfolder="unet",
        ignore_mismatched_sizes=True,  # 중요: class_embedding mismatch 무시
        low_cpu_mem_usage=False,
    )
    unet.load_state_dict(pretrained_unet.state_dict(), strict=False)

    # noise_scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    # def compute_snr(timesteps):
    #     """
    #     Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    #     """
    #     alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)

    #     sqrt_alphas_cumprod = alphas_cumprod**0.5
    #     sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    #     alpha = sqrt_alphas_cumprod[timesteps]
    #     sigma = sqrt_one_minus_alphas_cumprod[timesteps]
    #     snr = (alpha / sigma) ** 2
    #     return snr

    # tokenizer = RobertaTokenizerFast.from_pretrained(
    # base_model_id, subfolder="tokenizer")

    # text_encoder = ClapTextModelWithProjection.from_pretrained(
    #     base_model_id, subfolder="text_encoder"
    # )

    # vae = AutoencoderKL.from_pretrained(
    #     base_model_id, subfolder="vae"
    # )

    # pipe = AudioLDMPipeline.from_pretrained(base_model_id)
    # unet_config = pipe.unet.config

    # # 2. class embedding 관련 설정 제거
    # unet_config.num_class_embeds = 0
    # unet_config.class_embed_type = None

    # # 3. 새 UNet 모델 생성
    # unet = UNet2DConditionModel.from_config(unet_config)

    # # 2. class embedding 제거 (config 수정)
    # unet_config["num_class_embeds"] = 0
    # unet_config["class_embed_type"] = None

    # # 3. 수정된 config로 UNet 초기화
    # unet = UNet2DConditionModel(**unet_config)

    # pretrained_unet = UNet2DConditionModel.from_pretrained(
    #     base_model_id,
    #     subfolder="unet",
    #     num_class_embeds=0,
    #     class_embed_type=None,
    #     ignore_mismatched_sizes=True,
    #     low_cpu_mem_usage=False
    # )

    # # strict=False로 로드해서 일부 누락된 파라미터 무시
    # unet.load_state_dict(pretrained_unet.state_dict(), strict=False)
    
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
    # mel 변환기: 흑백 이미지 -> [1, H, W] Tensor
    mel_transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W) → (1, H, W), float32로 변환
    ])
    # 데이터셋 로드 코드 작성
    def collate_fn(examples):
        # MEL 이미지 변환
        mel_tensors = [mel_transform(example["mel"]) for example in examples]
        mel = torch.stack(mel_tensors)

        # 캡션 tokenizer 적용
        captions = [example["caption"] for example in examples]
        tokenized = tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt")

        return {
            "mel": mel,
            "caption": tokenized.input_ids,
        }


    
    dataset_name = "deetsadi/musiccaps_spectrograms"
    dataset = load_dataset(
        dataset_name
    )

    filtered_dataset = dataset["train"].filter(
    lambda example: "hiphop" in example["caption"].lower()
    )

    ###

    ### Train
    output_dir = "AudioLDM-with-LoRA/data/LoRA_weight"

    train_batch_size = 16
    total_batch_size = train_batch_size * accelerator.num_processes * accelerator.num_processes
    num_train_epochs = 100
    gradient_accumulation_steps = 1
    max_train_steps = 15000

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(filtered_dataset)}")
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

    num_workers = 0
    train_dataloader = torch.utils.data.DataLoader(
        filtered_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=num_workers,
    )


    for epoch in range(first_epoch, num_train_epochs):
        # vae.train()
        unet.train()

        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["mel"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
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

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["caption"], return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                target = noise
                class_labels = torch.zeros(step, unet.class_embedding.in_features).to(noisy_latents.dtype).to(latents.device)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, class_labels=class_labels, return_dict=False)[0]

                # Predict the noise residual and compute loss
                # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
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

            if global_step >= args.max_train_steps:
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