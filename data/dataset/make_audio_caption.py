import os
from datasets import load_dataset, Audio
import soundfile as sf

# 1. 데이터셋 로드 및 필터링
dataset = load_dataset("AndreiBlahovici/LP-MusicCaps-MTT", split="valid")

# "hip hop" 포함 + "low quality" 미포함 필터링
filtered_dataset = dataset.filter(lambda example: "hip hop" in example["genre"].lower())
# filtered_dataset = filtered_dataset.filter(lambda example: "low quality" not in example["gt_caption"].lower())

# 2. 저장 디렉터리 설정
base_dir = "/home/2020112030/WorkSpace/2025/AudioLDM-with-LoRA/data/dataset/out"
output_audio_dir = os.path.join(base_dir, "wavs_test")
output_caption_dir = os.path.join(base_dir, "captions_test")
os.makedirs(output_audio_dir, exist_ok=True)
os.makedirs(output_caption_dir, exist_ok=True)

# 3. 데이터셋 내의 오디오를 wav로, 캡션을 txt로 저장
for idx, example in enumerate(filtered_dataset):
    # 오디오 저장 (.wav)
    audio = example["audio"]  # dict with keys: 'array', 'sampling_rate'
    wav_path = os.path.join(output_audio_dir, f"audio_{idx:05d}.wav")
    sf.write(wav_path, audio["array"], audio["sampling_rate"])
    
    # 캡션 저장 (.txt)
    caption = example["gt_caption"]
    caption_path = os.path.join(output_caption_dir, f"audio_{idx:05d}.txt")
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption)

print(f"✅ 저장 완료: {len(filtered_dataset)}개 샘플")
