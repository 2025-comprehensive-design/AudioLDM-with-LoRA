import os
import pandas as pd
import torch, torchaudio
from torch.utils.data import Dataset

import yaml

def load_config(config_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "..", "..", config_path)  # script/utilities/ 기준
    config_path = os.path.normpath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

# dataset config
config = load_config("config/datasetConfig.yaml")

# 데이터셋 샘플링(musiccaps를 로컬에 다운로드 및 해당 오디오를 다운로드하여 사용함)
class MusicCapsDataset(Dataset):
    def __init__(self, csv_path, audio_dir, max_samples=config["max_train_steps"], target_duration=10, sampling_rate=48000):
        # csv 파일의 max_samples만큼 사용
        self.data = pd.read_csv(csv_path).head(max_samples)
        self.audio_dir = audio_dir              # 오디오 파일 경로
        self.target_duration = target_duration  # 10초(오디오 길이)
        self.sampling_rate = sampling_rate      # 오디오 샘플링 레이트(초당 쪼개는 정도)
        self.target_length = self.target_duration * self.sampling_rate  # 오디오 샘플 수

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # idx에 해당하는 row 가져옴
        row = self.data.iloc[idx]
        # 유튜브 id, 캡션, 시작 시간 가져옴
        ytid = row["ytid"]
        caption = row["caption"]
        start_s = row.get("start_s", 0)

        audio_path = os.path.join(self.audio_dir, f"{ytid}.wav")
        # 오디오 파일 로드
        audio, sr = torchaudio.load(audio_path)
        audio = audio[0]  # mono 채널만

        # 샘플링 레이트가 다르면 리샘플링
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            audio = resampler(audio)

        # 오디오 길이 맞추기
        start_sample = int(start_s * self.sampling_rate)
        end_sample = start_sample + self.target_length

        # 오디오 길이가 target_length보다 짧으면 패딩
        if end_sample > audio.shape[0]:
            audio = torch.nn.functional.pad(audio, (0, end_sample - audio.shape[0]))

        # 시작 위치부터 10초까지 사용
        audio = audio[start_sample:end_sample]

        return audio, caption