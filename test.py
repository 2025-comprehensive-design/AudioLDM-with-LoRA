from diffusers import AudioLDMPipeline

from datasets import load_dataset, DatasetDict
from script.data.datasets import HfAudioDataset
import torch

base_model_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(base_model_id)
# noise_scheduler = pipe.scheduler
# tokenizer = pipe.tokenizer
# unet = pipe.unet
# text_encoder = pipe.text_encoder
# vae = pipe.vae
validation_prompt = "hiphop"
output = pipe(validation_prompt, num_inference_steps=30)
print(output)

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
# unet.print_trainable_parameters()

# dataset = load_dataset("mb23/music_caps_4sec_wave_type_classical", split="train")
# train_dataset = HfAudioDataset(dataset)
# train_dataloader = torch.utils.data.DataLoader(
#         train_dataset,
#         shuffle=True,
#         batch_size=8,
#         num_workers=16,
#     )

# for batch in train_dataloader:
#     print("🔎 Batch Keys:", batch.keys())
#     print("📐 log_mel_spec shape:", batch["log_mel_spec"].shape)
#     print("📐 stft shape:", batch["stft"].shape)
#     print("📐 waveform shape:", batch["waveform"].shape)
#     print("📝 caption (text):", batch["text"])
#     print("🏷️ label vector:", batch["label_vector"])
#     break  # 하나만 확인하고 종료
