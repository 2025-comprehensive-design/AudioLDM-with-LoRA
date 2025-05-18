import os
from datasets import Dataset, Audio
from huggingface_hub import create_repo, HfApi
import soundfile as sf

# === 사용자 설정 ===
REPO_ID = "Rofla/AudioLDM-with-LoRA-Hiphop-subgenre"
AUDIO_DIR = "/home/2020112030/WorkSpace/2025/AudioLDM-with-LoRA/data/dataset/out/wavs"
CAPTION_DIR = "/home/2020112030/WorkSpace/2025/AudioLDM-with-LoRA/data/dataset/out/captions"

# === 데이터 리스트 생성 ===
data = []
file_names = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))

for file_name in file_names:
    idx = file_name.replace(".wav", "")
    wav_path = os.path.join(AUDIO_DIR, file_name)
    caption_path = os.path.join(CAPTION_DIR, f"{idx}.txt")

    if os.path.exists(caption_path):
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        data.append({
            "audio": wav_path,
            "caption": caption
        })

# === Dataset 생성 및 오디오 열 타입 지정 ===
dataset = Dataset.from_list(data)
dataset = dataset.cast_column("audio", Audio())

# === Dataset 업로드 ===
print(f"🔼 Hugging Face Hub로 업로드 중: {REPO_ID} ...")
dataset.push_to_hub(REPO_ID)
print("✅ 업로드 완료!")