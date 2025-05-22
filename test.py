import os
import librosa
import soundfile as sf

# 입력 파일 경로
input_wav_path = "/home/2020112030/WorkSpace/2025/AudioLDM-with-LoRA/data/dataset/out/wavs_test/audio_00003.wav"  # 자를 원본 .wav 파일 경로
output_dir = "/home/2020112030/WorkSpace/2025/AudioLDM-with-LoRA/data/dataset/out/wavs_test"          # 자른 파일을 저장할 디렉터리
os.makedirs(output_dir, exist_ok=True)

# 자를 길이 (초 단위)
segment_duration = 4.0

# 오디오 로딩
y, sr = librosa.load(input_wav_path, sr=None)  # sr=None이면 원본 샘플링레이트 유지
total_duration = librosa.get_duration(y=y, sr=sr)

# 자르기
segment_samples = int(segment_duration * sr)
num_segments = int(len(y) // segment_samples)

print(f"🎧 총 길이: {total_duration:.2f}초 / {num_segments}개의 4초 segment 생성 예정")

for i in range(num_segments):
    start_sample = i * segment_samples
    end_sample = start_sample + segment_samples
    segment = y[start_sample:end_sample]
    
    output_path = os.path.join(output_dir, f"segment_{i:03d}.wav")
    sf.write(output_path, segment, sr)
    print(f"✅ 저장: {output_path}")

print("✅ 전체 완료")
