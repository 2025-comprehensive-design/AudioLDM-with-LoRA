from diffusers import AudioLDMPipeline, StableDiffusionPipeline

from datasets import load_dataset, DatasetDict
from script.data.datasets import HfAudioDataset
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import ClapTextModelWithProjection, RobertaTokenizerFast, SpeechT5HifiGan
import scipy

base_model_id = "cvssp/audioldm-s-full-v2"
pipe = StableDiffusionPipeline.from_pretrained(base_model_id)

# noise_scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
# tokenizer = RobertaTokenizerFast.from_pretrained(base_model_id, subfolder="tokenizer")
# unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
# text_encoder = ClapTextModelWithProjection.from_pretrained(base_model_id, subfolder="text_encoder")
# vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
# vocoder = SpeechT5HifiGan.from_pretrained(base_model_id, subfolder="vocoder")
validation_prompt = "hiphop"

audio = pipe(validation_prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]

# save the audio sample as a .wav file
scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)



# print(unet.config.num_class_embeds)

import torch
from diffusers import UNet2DConditionModel

# unet_config = UNet2DConditionModel.load_config("cvssp/audioldm-s-full-v2", subfolder="unet")
# tokenizer = RobertaTokenizerFast.from_pretrained(base_model_id, subfolder="tokenizer")

# unet = UNet2DConditionModel.from_config(unet_config)

# print(f"unet.config : {unet.config} \n\n")
# print(f"vae.config : {vae.config}")


# batch_size = 4
# sample = torch.randn(batch_size, 8, 128, 128)
# timestep = torch.randint(0, 1000, (batch_size,))
# encoder_hidden_states = torch.randn(batch_size, 77, 768)
# class_labels = None

# try:
#     output = unet(sample, timestep, encoder_hidden_states, class_labels=class_labels, return_dict=False)[0]
#     print("Forward pass 성공")
# except ValueError as e:
#     print(f"ValueError 발생: {e}")


# unet.requires_grad_(False)
# vae.requires_grad_(False)
# text_encoder.requires_grad_(False)


# unet_lora_config = LoraConfig(
#         r=2,
#         lora_alpha=2,
#         init_lora_weights="gaussian",
#         target_modules=["to_k", "to_q", "to_v", "to_out.0"],
# )

# unet = get_peft_model(unet, unet_lora_config)
# # unet.print_trainable_parameters()
# def collate_fn(examples):
#     log_mel_spec = torch.stack([example["log_mel_spec"].unsqueeze(0) for example in examples])
#     input_ids = torch.stack([example["text"] for example in examples])
#     attention_mask = torch.stack([example["attention_mask"] for example in examples]) # attention mask 스택

#     return {"log_mel_spec": log_mel_spec, "input_ids": input_ids, "attention_mask" : attention_mask}

# dataset = load_dataset("mb23/music_caps_4sec_wave_type_classical", split="train")
# train_dataset = HfAudioDataset(dataset)
# train_dataloader = torch.utils.data.DataLoader(
#         train_dataset,
#         shuffle=True,
#         batch_size=8,
#         collate_fn=collate_fn,
#         num_workers=16,
#     )
