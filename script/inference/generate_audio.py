
import os
import torch
import soundfile as sf
from diffusers import AudioLDMPipeline, UNet2DConditionModel
from safetensors.torch import load_file

def main():
    base_model_id = "cvssp/audioldm-s-full-v2"
    lora_weights_path = "/home/2019111986/AudioLDM-LoRA-unet/AudioLDM-with-LoRA/data/LoRA_weight/checkpoint-20000/model.safetensors"
    prompt = "This instrumental track blends lively flute melodies together with punchy drums, delivering a unique listening experience that captivates the ear without the need for vocals. The subgenre of hip-hop is boom bap."

    checkpoint_name = os.path.basename(os.path.dirname(lora_weights_path))
    output_dir = f"./AudioLDM-with-LoRA/generated_audio/{checkpoint_name}"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = AudioLDMPipeline.from_pretrained(base_model_id)
    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    state_dict = load_file(lora_weights_path)
    unet.load_state_dict(state_dict, strict=False)
    pipe.unet = unet
    pipe = pipe.to(device)

    for i in range(3):
        audio = pipe(prompt=prompt, num_inference_steps=50, audio_length_in_s=4.0, guidance_scale=7.5).audios[0]
        output_path = os.path.join(output_dir, f"gen_{i}.wav")
        sf.write(output_path, audio, samplerate=16000)
        print(f"[✓] gen_{i}.wav saved")

if __name__ == "__main__":
    main()
