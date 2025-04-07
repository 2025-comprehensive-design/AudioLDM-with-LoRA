from diffusers import AudioLDMPipeline
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

base_model_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(base_model_id)
noise_scheduler = pipe.scheduler
tokenizer = pipe.tokenizer
unet = pipe.unet
text_encoder = pipe.text_encoder
vae = pipe.vae

unet.requires_grad_(False)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)


unet_lora_config = LoraConfig(
        r=2,
        lora_alpha=2,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)

unet = get_peft_model(unet, unet_lora_config)
unet.print_trainable_parameters()