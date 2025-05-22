import os
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
from random import sample
from diffusers import AudioLDMPipeline, UNet2DConditionModel
from safetensors.torch import load_file
from transformers import ClapModel, AutoProcessor
from peft import LoraConfig, get_peft_model_state_dict, get_peft_model
from diffusers.utils import convert_state_dict_to_diffusers
import matplotlib.pyplot as plt
import librosa.display

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)

def save_mel_spectrogram(audio, sr, save_path, title=None):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_clap_similarity(audio_waveform: np.ndarray, text: str, processor, model, device):
    inputs = processor(audios=audio_waveform, return_tensors="pt", sampling_rate=48000).to(device)
    with torch.no_grad():
        audio_embed = model.get_audio_features(**inputs)
        text_embed = model.get_text_features(**processor(text=text, return_tensors="pt", padding=True).to(device))
        audio_embed = F.normalize(audio_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        similarity = (audio_embed @ text_embed.T).item()
        return (similarity + 1) / 2

def median_pairwise_distance(x):
    return torch.median(torch.pdist(x)).item()

def calc_kernel_audio_distance(x, y, device="cuda", bandwidth=None):
    x = x.to(dtype=torch.float32, device=device)
    y = y.to(dtype=torch.float32, device=device)

    if bandwidth is None:
        bandwidth = median_pairwise_distance(y)
        if bandwidth < 1e-6 or torch.isnan(torch.tensor(bandwidth)):
            bandwidth = 1.0

    gamma = 1 / (2 * bandwidth**2 + 1e-8)
    kernel_fn = lambda a: torch.exp(-gamma * a)

    xx = x @ x.T
    x_sq = torch.diagonal(xx)
    d2_xx = x_sq.unsqueeze(1) + x_sq.unsqueeze(0) - 2 * xx
    k_xx = kernel_fn(d2_xx)
    k_xx -= torch.diag(torch.diagonal(k_xx))
    k_xx_mean = k_xx.sum() / (x.shape[0] * (x.shape[0] - 1))

    yy = y @ y.T
    y_sq = torch.diagonal(yy)
    d2_yy = y_sq.unsqueeze(1) + y_sq.unsqueeze(0) - 2 * yy
    k_yy = kernel_fn(d2_yy)
    k_yy -= torch.diag(torch.diagonal(k_yy))
    k_yy_mean = k_yy.sum() / (y.shape[0] * (y.shape[0] - 1))

    xy = x @ y.T
    d2_xy = x_sq.unsqueeze(1) + y_sq.unsqueeze(0) - 2 * xy
    k_xy = kernel_fn(d2_xy)
    k_xy_mean = k_xy.mean()

    raw_kad = (k_xx_mean + k_yy_mean - 2 * k_xy_mean).item()
    stabilized_kad = max(raw_kad, 0.0)

    return stabilized_kad

def compute_kad_score(gen_audios, ref_audios, model, processor, device):
    gen_embeds, ref_embeds = [], []
    for gen in gen_audios:
        gen_input = processor(audios=gen, return_tensors="pt", sampling_rate=48000).to(device)
        with torch.no_grad():
            gen_embed = model.get_audio_features(**gen_input)
        gen_embeds.append(F.normalize(gen_embed.squeeze(0), dim=-1))

    for ref in ref_audios:
        ref_input = processor(audios=ref, return_tensors="pt", sampling_rate=48000).to(device)
        with torch.no_grad():
            ref_embed = model.get_audio_features(**ref_input)
        ref_embeds.append(F.normalize(ref_embed.squeeze(0), dim=-1))

    gen_tensor = torch.stack(gen_embeds)
    ref_tensor = torch.stack(ref_embeds)
    return calc_kernel_audio_distance(gen_tensor, ref_tensor, device=device)

def generate_audio(pipe, prompt, num_samples, output_dir, prefix, weight):
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for i in range(1, num_samples + 1):
        audio = pipe(prompt=prompt, num_inference_steps=50, audio_length_in_s=4.0, guidance_scale=7.5).audios[0]
        audio = audio / np.max(np.abs(audio))

        if (prefix == "lora"):
            fname = f"{weight}_{prefix}_{i:03}.wav"
        else:
            fname = f"{prefix}_{i:03}.wav"
        fpath = os.path.join(output_dir, fname)
        sf.write(fpath, audio, samplerate=16000)
        print(f"[{prefix}] 오디오 저장 완료: {fname}")
        paths.append(fpath)

        img_path = fpath.replace(".wav", ".png")
        save_mel_spectrogram(audio, sr=16000, save_path=img_path, title=f"{prefix} {i:03}")

        audio_48k = librosa.resample(audio, orig_sr=16000, target_sr=48000)
        score = compute_clap_similarity(audio_48k, prompt, processor, clap_model, device)
        print(f"[{prefix}] {fname} CLAP Score: {score:.4f}")
    
    return paths

def load_and_resample(paths):
    audios = []
    for path in paths:
        audio, _ = librosa.load(path, sr=16000)
        audio_48k = librosa.resample(audio, orig_sr=16000, target_sr=48000)
        audios.append(audio_48k)
    return audios

def get_top3_audios(scores_and_paths):
    sorted_items = sorted(scores_and_paths, key=lambda x: x[0], reverse=True)[:3]
    audios = []
    paths = []
    for _, path in sorted_items:
        audio, _ = librosa.load(path, sr=16000)
        audio_48k = librosa.resample(audio, orig_sr=16000, target_sr=48000)
        audios.append(audio_48k)
        paths.append(path)
    return audios, paths

def get_random3_audios(all_paths, exclude_paths):
    available = [p for p in all_paths if p not in exclude_paths]
    selected = sample(available, 3)
    return load_and_resample(selected)

def main():
    weight = "97000-r4-a4"
    base_model_id = "cvssp/audioldm-s-full-v2"
    lora_weights_path = f"C:/Users/gkseo/audioldm/AudioLDM-with-LoRA/data/LoRA_weight/checkpoint-{weight}/model.safetensors"
    prompt = "This instrumental track blends lively flute melodies together with punchy drums, delivering a unique listening experience that captivates the ear without the need for vocals. The subgenre of hip-hop is boom bap"
    # prompt = "A male singer sings this cool hip hop love song with backup singers in vocal harmony. The tempo is medium with keyboard accompaniment, piano accompaniment, steady drumming rhythm, clapping percussion and other sonic effects. The song is emotional and romantic with a cool dance  groove, The subgenre of hip-hop is New jack swing"

    base_pipe = AudioLDMPipeline.from_pretrained(base_model_id).to(device)

    lora_pipe = AudioLDMPipeline.from_pretrained(base_model_id)
    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")

    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=["to_q", "to_v"],
        init_lora_weights="gaussian"
    )
    unet_lora = get_peft_model(unet, lora_config)

    lora_state = load_file(lora_weights_path)
    unet_lora.load_state_dict(lora_state, strict=False)
    lora_diffusers_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet_lora))
    unet.load_state_dict(lora_diffusers_state, strict=False)

    print("LoRA 적용 레이어:")
    for name, param in unet.named_parameters():
        if "lora" in name:
            print(name, param.shape)
    lora_pipe = AudioLDMPipeline.from_pretrained(base_model_id, unet=unet).to(device)

    base_paths = generate_audio(base_pipe, prompt, num_samples=10, output_dir="./base_audio", prefix="base", weight=weight)
    lora_paths = generate_audio(lora_pipe, prompt, num_samples=10, output_dir="./lora_audio", prefix="lora", weight=weight)

    base_scores_and_paths = []
    for path in base_paths:
        audio, _ = librosa.load(path, sr=16000)
        audio_48k = librosa.resample(audio, orig_sr=16000, target_sr=48000)
        score = compute_clap_similarity(audio_48k, prompt, processor, clap_model, device)
        base_scores_and_paths.append((score, path))

    lora_scores_and_paths = []
    for path in lora_paths:
        audio, _ = librosa.load(path, sr=16000)
        audio_48k = librosa.resample(audio, orig_sr=16000, target_sr=48000)
        score = compute_clap_similarity(audio_48k, prompt, processor, clap_model, device)
        lora_scores_and_paths.append((score, path))

    base_top3, base_top3_paths = get_top3_audios(base_scores_and_paths)
    lora_top3, lora_top3_paths = get_top3_audios(lora_scores_and_paths)

    base_all_paths = [path for _, path in base_scores_and_paths]
    lora_all_paths = [path for _, path in lora_scores_and_paths]

    base_random3 = get_random3_audios(base_all_paths, base_top3_paths)
    lora_random3 = get_random3_audios(lora_all_paths, lora_top3_paths)

    # refefence 오디오
    ref_dir = "./ref_audio"
    ref_files = ["segment_000.wav", "segment_001.wav", "segment_002.wav"]
    ref_paths = [os.path.join(ref_dir, f) for f in ref_files]
    ref_audios = load_and_resample(ref_paths)

    # 학습 데이터셋의 ref 오디오
    # ref_dir = "./downloaded_audio"
    # ref_files = ["track_1.wav", "track_2.wav", "track_3.wav"]
    # ref_paths = [os.path.join(ref_dir, f) for f in ref_files]
    # ref_audios = load_and_resample(ref_paths)

    kad_base_top3_vs_lora_top3 = compute_kad_score(base_top3, lora_top3, clap_model, processor, device)
    kad_base_top3_vs_base_random = compute_kad_score(base_top3, base_random3, clap_model, processor, device)
    kad_lora_top3_vs_lora_random = compute_kad_score(lora_top3, lora_random3, clap_model, processor, device)
    kad_base_top3_vs_ref = compute_kad_score(base_top3, ref_audios, clap_model, processor, device)
    kad_lora_top3_vs_ref = compute_kad_score(lora_top3, ref_audios, clap_model, processor, device)

    print("\nKAD 비교 결과:")
    print(f"Base Top-3 vs LoRA Top-3         : {kad_base_top3_vs_lora_top3:.4f}")
    print(f"Base Top-3 vs Base Random 3      : {kad_base_top3_vs_base_random:.4f}")
    print(f"LoRA Top-3 vs LoRA Random 3      : {kad_lora_top3_vs_lora_random:.4f}")
    print(f"Base Top-3 vs Reference          : {kad_base_top3_vs_ref:.4f}")
    print(f"LoRA Top-3 vs Reference          : {kad_lora_top3_vs_ref:.4f}")

if __name__ == "__main__":
    main()
