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

import logging
import numpy as np
from contextlib import nullcontext

import torch, random
import torch.nn.functional as F
import torch.nn as nn

import torchaudio
from torchaudio import transforms as AT
from torchvision import transforms as IT


from script.data.datasets import HfAudioDataset
from datasets import load_dataset

# LoRA
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
    
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from diffusers import AudioLDMPipeline
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version, is_wandb_available, convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.optimization import get_scheduler
from transformers import ClapTextModelWithProjection, RobertaTokenizerFast, SpeechT5HifiGan
from transformers import AutoProcessor, ClapModel

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

import librosa
import librosa.display
import io
from PIL import Image

import torch, gc

gc.collect()
torch.cuda.empty_cache()

logger = get_logger(__name__)

# 시각화 툴

base_model_id = "cvssp/audioldm-s-full-v2"
dataset_hub_id = "Rofla/AudioLDM-with-LoRA-Hiphop-subgenre"
validation_prompt = "hip hop music, The subgenre of hip-hop is boom bap."
validation_epochs = 100
SCALE_FACTOR = 100

if is_wandb_available():
    import wandb

num_validation_images = 5

def plot_spectrogram_to_image(spec, title=None):
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
    original_pipeline=None,
    clap_model=None,
    clap_processor=None,
    ref_audios=None
):
    logger.info(
        f"Running validation... \n Generating {num_validation_images} mel images with prompt:"
        f" {validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    if original_pipeline:
        original_pipeline = original_pipeline.to(accelerator.device)
        original_pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device)
    images, mel_spectrogram_images = [], []
    orig_images, orig_mel_spectrogram_images = [], []
    clap_scores, original_clap_scores = [], []
    kad_score_lora, kad_score_original=[],[]

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    def compute_clap_similarity(audio_waveform: np.ndarray, text: str) -> float:
        inputs = clap_processor(audios=audio_waveform, return_tensors="pt", sampling_rate=48000)
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        with torch.no_grad():
            audio_embed = clap_model.get_audio_features(**inputs)
            text_embed = clap_model.get_text_features(**clap_processor(text=text, return_tensors="pt", padding=True).to(accelerator.device))
            audio_embed = F.normalize(audio_embed, dim=-1)
            text_embed = F.normalize(text_embed, dim=-1)
            similarity = (audio_embed @ text_embed.T).item()
            return (similarity + 1) / 2

    with autocast_ctx:
        for i in range(num_validation_images):
            # 파인튜닝 모델 출력
            audio_output = pipeline(validation_prompt, num_inference_steps=50, generator=generator, audio_length_in_s=4.0)
            audio = audio_output.audios[0]
            images.append(audio)

            mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=1024, hop_length=512, n_mels=64)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            spec_image = plot_spectrogram_to_image(mel_spec_db, title=f"Spectrogram {i}: {validation_prompt}")
            mel_spectrogram_images.append(spec_image)

            # CLAP 점수 계산 (파인튜닝 모델)
            if clap_model and clap_processor:
                print(f"# CLAP 점수 계산 (파인튜닝 모델)")
                # 48000Hz로 resample
                resampled_audio = librosa.resample(audio, orig_sr=16000, target_sr=48000)
                clap_score = compute_clap_similarity(resampled_audio, validation_prompt)
                clap_scores.append(clap_score)

            # 원본 모델 출력
            if original_pipeline:
                orig_output = original_pipeline(validation_prompt, num_inference_steps=50, generator=generator, audio_length_in_s=4.0)
                orig_audio = orig_output.audios[0]
                orig_images.append(orig_audio)

                orig_mel_spec = librosa.feature.melspectrogram(y=orig_audio, sr=16000, n_fft=1024, hop_length=512, n_mels=64)
                orig_mel_spec_db = librosa.power_to_db(orig_mel_spec, ref=np.max)
                orig_spec_image = plot_spectrogram_to_image(orig_mel_spec_db, title=f"Original Spectrogram {i}: {validation_prompt}")
                orig_mel_spectrogram_images.append(orig_spec_image)

                if clap_model and clap_processor:
                    print(f"# CLAP 점수 계산 (원본 모델)")
                    resampled_orig_audio = librosa.resample(orig_audio, orig_sr=16000, target_sr=48000)
                    original_clap_score = compute_clap_similarity(resampled_orig_audio, validation_prompt)
                    original_clap_scores.append(original_clap_score)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"

        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
            if original_pipeline:
                orig_np_images = np.stack([np.asarray(img) for img in orig_images])
                tracker.writer.add_images(f"original_{phase_name}", orig_np_images, epoch, dataformats="NHWC")

        if tracker.name == "wandb":
            audio_logs = [wandb.Audio(img, sample_rate=16000, caption=f"{i}: {validation_prompt}") for i, img in enumerate(images)]
            tracker.log({phase_name: audio_logs})

            spec_logs = [wandb.Image(img, caption=f"{phase_name} Spectrogram {i}: {validation_prompt}") for i, img in enumerate(mel_spectrogram_images)]
            tracker.log({f"{phase_name}_spectrogram": spec_logs})

            if original_pipeline:
                orig_audio_logs = [wandb.Audio(img, sample_rate=16000, caption=f"Original {i}: {validation_prompt}") for i, img in enumerate(orig_images)]
                tracker.log({f"original_{phase_name}": orig_audio_logs})

                orig_spec_logs = [wandb.Image(img, caption=f"Original {phase_name} Spectrogram {i}: {validation_prompt}") for i, img in enumerate(orig_mel_spectrogram_images)]
                tracker.log({f"original_{phase_name}_spectrogram": orig_spec_logs})

            # wandb 로그
            if accelerator.is_main_process:
                print("*********TRUE!*********")
                if clap_scores:
                    avg_clap_score = np.mean(clap_scores)
                    wandb.log({f"{phase_name}_clap_score": avg_clap_score}, step=epoch + 10)
                    # tracker.log({f"{phase_name}_clap_score": avg_clap_score}, step=epoch)
                if original_clap_scores:
                    avg_original_clap_score = np.mean(original_clap_scores)
                    wandb.log({f"original_{phase_name}_clap_score": avg_original_clap_score}, step=epoch + 10)
                    # tracker.log({f"original_{phase_name}_clap_score": avg_original_clap_score}, step=epoch)

                if len(images) == len(orig_images):
                    kad_score_lora = compute_clap_kad_from_audio_lists(
                    ref_audios=ref_audios,
                    gen_audios=images,
                    clap_model=clap_model,
                    clap_processor=clap_processor,
                    device=accelerator.device
                    )

                    kad_score_original = compute_clap_kad_from_audio_lists(
                    ref_audios=ref_audios,
                    gen_audios=orig_images,
                    clap_model=clap_model,
                    clap_processor=clap_processor,
                    device=accelerator.device)

                    wandb.log({f"{phase_name}_kad_score_lora": kad_score_lora,
                               f"{phase_name}_kad_score_original": kad_score_original}, step=epoch + 10)

    return images, avg_clap_score, avg_original_clap_score, kad_score_lora, kad_score_original


def median_pairwise_distance(x, subsample=None):
    x = torch.tensor(x, dtype=torch.float32)
    n_samples = x.shape[0]
    if subsample is not None and subsample < n_samples * (n_samples - 1) / 2:
        idx1 = torch.randint(0, n_samples, (subsample,))
        idx2 = torch.randint(0, n_samples, (subsample,))
        mask = idx1 == idx2
        idx2[mask] = (idx2[mask] + 1) % n_samples
        distances = torch.sqrt(torch.sum((x[idx1] - x[idx2])**2, dim=1))
    else:
        distances = torch.pdist(x)
    return torch.median(distances).item()

def calc_kernel_audio_distance(x, y, device="cuda", bandwidth=None, kernel='gaussian', eps=1e-8):
    x = x.to(dtype=torch.float32, device=device)
    y = y.to(dtype=torch.float32, device=device)

    print(f"[KAD] x shape: {x.shape}, y shape: {y.shape}")
    print(f"[KAD] x norm: {torch.norm(x, dim=1)}")
    print(f"[KAD] y norm: {torch.norm(y, dim=1)}")

    if bandwidth is None:
        bandwidth = median_pairwise_distance(y)
        if bandwidth < 1e-6 or torch.isnan(torch.tensor(bandwidth)):
            print(f"[KAD] Warning: bandwidth too small or NaN, fallback to 1.0")
            bandwidth = 1.0  # 기본값 보정

    gamma = 1 / (2 * bandwidth**2 + eps)
    if kernel == 'gaussian':
        kernel_fn = lambda a: torch.exp(-gamma * a)
    elif kernel == 'iq':
        kernel_fn = lambda a: 1 / (1 + gamma * a)
    elif kernel == 'imq':
        kernel_fn = lambda a: 1 / torch.sqrt(1 + gamma * a)
    else:
        raise ValueError("Invalid kernel type")

    # x-x
    xx = x @ x.T
    x_sqnorms = torch.diagonal(xx)
    d2_xx = x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0) - 2 * xx
    k_xx = kernel_fn(d2_xx)
    k_xx = k_xx - torch.diag(torch.diagonal(k_xx))
    k_xx_mean = k_xx.sum() / (x.shape[0] * (x.shape[0] - 1))

    # y-y
    yy = y @ y.T
    y_sqnorms = torch.diagonal(yy)
    d2_yy = y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * yy
    k_yy = kernel_fn(d2_yy)
    k_yy = k_yy - torch.diag(torch.diagonal(k_yy))
    k_yy_mean = k_yy.sum() / (y.shape[0] * (y.shape[0] - 1))

    # x-y
    xy = x @ y.T
    d2_xy = x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * xy
    k_xy = kernel_fn(d2_xy)
    k_xy_mean = k_xy.mean()

    result = k_xx_mean + k_yy_mean - 2 * k_xy_mean
    return result * SCALE_FACTOR

def compute_clap_kad_from_audio_lists(ref_audios, gen_audios, clap_model, clap_processor, device="cuda"):
    ref_embeddings = []
    gen_embeddings = []

    for idx, (ref_audio, gen_audio) in enumerate(zip(ref_audios, gen_audios)):
        ref_audio_rs = librosa.resample(ref_audio, orig_sr=16000, target_sr=48000)
        gen_audio_rs = librosa.resample(gen_audio, orig_sr=16000, target_sr=48000)

        ref_inputs = clap_processor(audios=ref_audio_rs, return_tensors="pt", sampling_rate=48000).to(device)
        gen_inputs = clap_processor(audios=gen_audio_rs, return_tensors="pt", sampling_rate=48000).to(device)

        with torch.no_grad():
            ref_embed = clap_model.get_audio_features(**ref_inputs)
            gen_embed = clap_model.get_audio_features(**gen_inputs)

            ref_embed = F.normalize(ref_embed, dim=-1)
            gen_embed = F.normalize(gen_embed, dim=-1)

        ref_embeddings.append(ref_embed.squeeze(0))
        gen_embeddings.append(gen_embed.squeeze(0))

    ref_tensor = torch.stack(ref_embeddings)  # [B, D]
    gen_tensor = torch.stack(gen_embeddings)  # [B, D]

    kad_score = calc_kernel_audio_distance(ref_tensor, gen_tensor, device=device)
    return kad_score.item()


def main() :
    accelerator_project_config = ProjectConfiguration(project_dir="../../", logging_dir="AudioLDM-with-LoRA/log")

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
                "tags": ["lora", "audioldm", "subgenre"],
                "name": "<task : r = 2, alpha = 2>"
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

    processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(accelerator.device)

    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    pipe = AudioLDMPipeline.from_pretrained(base_model_id, unet=unet)

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
        lora_alpha=2,
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
        lr=1.0e-5,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
        eps=1e-08,
    )

    num_workers = 4
    train_batch_size = 2
    total_batch_size = train_batch_size * accelerator.num_processes
    num_train_epochs = 1000
    gradient_accumulation_steps = 1
    max_train_steps = 97000
    checkpointing_steps = 9700 * 2
    total_train_loss = 0.0
    total_steps = 0

    def collate_fn(examples):
        log_mel_spec = torch.stack([example["log_mel_spec"].unsqueeze(0) for example in examples])
        input_ids = torch.stack([example["text"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])

        return {"log_mel_spec": log_mel_spec, "input_ids": input_ids, "attention_mask" : attention_mask}

    dataset = load_dataset(dataset_hub_id, split="train")
    train_dataset = HfAudioDataset(dataset)

    ref_audios = [librosa.util.fix_length(example["audio"]["array"], size=16000*10) for example in dataset.select(range(num_validation_images))
]


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
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    output_dir = os.path.join(project_root, "data", "LoRA_weight", "r2_alpha2")

    os.makedirs(output_dir, exist_ok=True)
    
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

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    avg_clap_score = 0.0
    avg_original_clap_score = 0.0
    kad_score_lora = 0.0
    kad_score_original = 0.0
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        optimizer.zero_grad()

        train_loss = 0.0
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
                
                prompt_bsz = prompt_embeds.shape[0]
                latents_bsz = batch["log_mel_spec"].shape[0]

                if prompt_bsz != latents_bsz:
                    repeat_factor = latents_bsz // prompt_bsz
                    prompt_embeds = prompt_embeds.repeat_interleave(repeat_factor, dim=0)

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=None,
                    class_labels=prompt_embeds,
                    cross_attention_kwargs={"scale": 1.0},
                    return_dict=False
                )[0]

                # 직접적인 loss 계산
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                avg_loss = accelerator.gather(loss).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps
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
            accelerator.log({
                                "avg_lora_clap_score": avg_clap_score,
                                "avg_original_clap_score": avg_original_clap_score,
                                "kad_score_lora": kad_score_lora,
                                "kad_score_original": kad_score_original
                            }, step=global_step)
            
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        if accelerator.is_main_process and validation_prompt is not None and epoch % validation_epochs == 0:
            unwrapped_unet = unwrap_model(unet)
            pipeline = AudioLDMPipeline.from_pretrained(base_model_id, unet=unwrapped_unet, torch_dtype=weight_dtype)
            images, avg_clap_score_A, avg_original_clap_score_A, kad_score_lora_A, kad_score_original_A = log_validation(pipeline, accelerator, epoch,original_pipeline=pipe,
                clap_model=clap_model,
                clap_processor=processor, ref_audios=ref_audios
            )
            avg_clap_score = float(avg_clap_score_A)
            avg_original_clap_score = float(avg_original_clap_score_A)
            kad_score_lora = float(kad_score_lora_A)
            kad_score_original = float(kad_score_original_A)
            del pipeline

            torch.cuda.empty_cache()

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

        if validation_prompt is not None:
            pipeline = AudioLDMPipeline.from_pretrained(base_model_id, unet=unet_lora_state_dict, torch_dtype=weight_dtype)
            images, avg_clap_score_A, avg_original_clap_score_A, kad_score_lora_A, kad_score_original_A = log_validation(pipeline, accelerator, epoch, is_final_validation=True, original_pipeline=pipe,
                clap_model=clap_model,
                clap_processor=processor, ref_audios=ref_audios
            )

    accelerator.end_training()

if __name__ == "__main__":
    main()