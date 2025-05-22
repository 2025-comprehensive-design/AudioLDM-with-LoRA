import os
import librosa
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoProcessor, ClapModel

def compute_clap_score(audio, text, clap_model, clap_processor, device):
    inputs = clap_processor(audios=audio, return_tensors="pt", sampling_rate=48000).to(device)
    with torch.no_grad():
        audio_embed = clap_model.get_audio_features(**inputs)
        text_inputs = clap_processor(text=text, return_tensors="pt", padding=True).to(device)
        text_embed = clap_model.get_text_features(**text_inputs)
        audio_embed = F.normalize(audio_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        score = (audio_embed @ text_embed.T).item()
        return (score + 1) / 2

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

    return (k_xx_mean + k_yy_mean - 2 * k_xy_mean).item()

def compute_kad_score(gen_audios, ref_audios, clap_model, clap_processor, device, bandwidth=1):
    gen_embeds, ref_embeds = [], []

    for gen, ref in zip(gen_audios, ref_audios):
        gen_rs = librosa.resample(gen, orig_sr=16000, target_sr=48000)
        ref_rs = librosa.resample(ref, orig_sr=16000, target_sr=48000)

        gen_input = clap_processor(audios=gen_rs, return_tensors="pt", sampling_rate=48000).to(device)
        ref_input = clap_processor(audios=ref_rs, return_tensors="pt", sampling_rate=48000).to(device)

        with torch.no_grad():
            gen_embed = clap_model.get_audio_features(**gen_input)
            ref_embed = clap_model.get_audio_features(**ref_input)

        gen_embeds.append(F.normalize(gen_embed.squeeze(0), dim=-1))
        ref_embeds.append(F.normalize(ref_embed.squeeze(0), dim=-1))

    gen_tensor = torch.stack(gen_embeds)
    ref_tensor = torch.stack(ref_embeds)

    return calc_kernel_audio_distance(gen_tensor, ref_tensor, device=device, bandwidth=bandwidth)

def main():
    prompt = "This instrumental track blends lively flute melodies together with punchy drums, delivering a unique listening experience that captivates the ear without the need for vocals. The subgenre of hip-hop is boom bap."
    gen_dir = "/home/2019111986/AudioLDM-LoRA-unet/AudioLDM-with-LoRA/generated_audio/checkpoint-20000"
    ref_dir = "/home/2019111986/AudioLDM-LoRA-unet/script/inference/kad_music"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLAP model
    clap_processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)

    def load_audios_from_dir(directory, label):
        audios = []
        for fname in sorted(os.listdir(directory)):
            if fname.endswith(".wav"):
                path = os.path.join(directory, fname)
                audio, _ = librosa.load(path, sr=16000)
                audios.append(audio)
                print(f"Loaded {label} file: {fname}")
        return audios

    gen_audios = load_audios_from_dir(gen_dir, "generated")
    ref_audios = load_audios_from_dir(ref_dir, "reference")

    # CLAP score
    print("\nCLAP Scores (Prompt vs. Generated):")
    clap_scores = []
    for i, audio in enumerate(gen_audios):
        resampled_audio = librosa.resample(audio, orig_sr=16000, target_sr=48000)
        score = compute_clap_score(resampled_audio, prompt, clap_model, clap_processor, device)
        clap_scores.append(score)
        print(f" - gen_{i}.wav | CLAP score: {score:.4f}")
    avg_clap = sum(clap_scores) / len(clap_scores)
    print(f"\nAverage CLAP Score: {avg_clap:.4f}")

    # KAD score
    kad_score = compute_kad_score(gen_audios, ref_audios, clap_model, clap_processor, device, bandwidth=1)
    print(f"\nKAD Score (Generated vs. Reference): {kad_score:.4f}")

if __name__ == "__main__":
    main()
