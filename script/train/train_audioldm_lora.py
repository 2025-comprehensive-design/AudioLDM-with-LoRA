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
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from diffusers import AudioLDMPipeline
# from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.optimization import get_scheduler

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

num_validation_images = 4

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
            tracker.log(
                {
                    phase_name: [
                        wandb.Audio(image, caption=f"{i}: {validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )
    return images



def main() :

    accelerator_project_config = ProjectConfiguration(project_dir="./", logging_dir="AudioLDM-with-LoRA/log")

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=None,
        log_with="wandb",
        project_config=accelerator_project_config,
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
    # def preprocess_train(example):
    #     mel = waveform_to_mel_tensor(torch.tensor(example["audio"]["array"]),
    #                                 sampling_rate=example["audio"]["sampling_rate"])
        
    #     # tokenizer 결과 전체 반환
    #     tokenized = tokenize_captions({"caption": [example["caption"]]})  # ❗ dict
    #     return {
    #         "audio": mel,
    #         "caption": {k: v[0] for k, v in tokenized.items()}  # 각 key에서 첫 번째 값만 추출 (1개 샘플이므로)
    #     }

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
    
    num_workers = 0
    train_batch_size = 16
    total_batch_size = train_batch_size * accelerator.num_processes * accelerator.num_processes
    num_train_epochs = 10
    gradient_accumulation_steps = 1
    max_train_steps = 15000
    checkpointing_steps = 500
    
    dataset = load_dataset("mb23/music_caps_4sec_wave_type_classical", split="train")

    # caption에 "guitar" 들어간 것만 필터링
    filtered_dataset = dataset.filter(
        lambda example: "guitar" in example["caption"].lower()
    )
    train_dataset = HfAudioDataset(filtered_dataset)

    # 필터링된 데이터셋으로 학습!
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
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

                # Convert images to latent space
                latents = vae.encode(batch["log_mel_spec"].unsqueeze(1).to(dtype=weight_dtype)).latent_dist.sample()
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
                ###
                text_inputs = tokenizer(
                    batch["text"],
                    padding="max_length",          # 또는 padding=True
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt"
                ).to(latents.device)
                
                encoder_hidden_states = text_encoder(
                    input_ids=text_inputs["input_ids"],
                    return_dict=False
                )[0]


                # encoder_hidden_states = text_encoder(batch["text"], return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                target = noise

                # Predict the noise residual and compute loss
                # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                label_vector = batch.get("label_vector", None)
                if isinstance(label_vector, torch.Tensor) and label_vector.ndim == 2 and label_vector.shape[1] > 0:
                    class_labels = torch.argmax(label_vector, dim=1)
                else:
                    continue

                model_pred = unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    class_labels=class_labels,
                    return_dict=False
                )[0]


                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # if args.snr_gamma is None:
                #     loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # else:
                #     # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                #     # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                #     # This is discussed in Section 4.2 of the same paper.
                #     snr = compute_snr(noise_scheduler, timesteps)
                #     mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                #         dim=1
                #     )[0]
                #     if noise_scheduler.config.prediction_type == "epsilon":
                #         mse_loss_weights = mse_loss_weights / snr
                #     elif noise_scheduler.config.prediction_type == "v_prediction":
                #         mse_loss_weights = mse_loss_weights / (snr + 1)

                #     loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                #     loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                #     loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # if global_step % args.checkpointing_steps == 0:
                #     if accelerator.is_main_process:
                #         # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                #         if args.checkpoints_total_limit is not None:
                #             checkpoints = os.listdir(args.output_dir)
                #             checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                #             checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                #             # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                #             if len(checkpoints) >= args.checkpoints_total_limit:
                #                 num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                #                 removing_checkpoints = checkpoints[0:num_to_remove]

                #                 logger.info(
                #                     f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                #                 )
                #                 logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                #                 for removing_checkpoint in removing_checkpoints:
                #                     removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                #                     shutil.rmtree(removing_checkpoint)

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
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
            if validation_prompt is not None and epoch % validation_epochs == 0:
                # create pipeline
                pipeline = AudioLDMPipeline.from_pretrained(
                    base_model_id,
                    unet=unwrap_model(unet),
                    torch_dtype=weight_dtype,
                )

                images = log_validation(pipeline, accelerator, epoch)

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
        if validation_prompt is not None:
            pipeline = AudioLDMPipeline.from_pretrained(
                base_model_id,
                torch_dtype=weight_dtype,
            )

            # load attention processors
            pipeline.load_lora_weights(output_dir)

            # run inference
            images = log_validation(pipeline, accelerator, epoch, is_final_validation=True)

        # if args.push_to_hub:
        #     save_model_card(
        #         repo_id,
        #         images=images,
        #         base_model=args.pretrained_model_name_or_path,
        #         dataset_name=args.dataset_name,
        #         repo_folder=args.output_dir,
        #     )
        #     upload_folder(
        #         repo_id=repo_id,
        #         folder_path=args.output_dir,
        #         commit_message="End of training",
        #         ignore_patterns=["step_*", "epoch_*"],
        #     )

    accelerator.end_training()

if __name__ == "__main__":
    main()