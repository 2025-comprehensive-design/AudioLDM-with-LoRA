import sys
import torch, random, torchaudio, os, json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from diffusers import AudioLDMPipeline
from peft import LoraConfig, get_peft_model
import wandb  
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration  
import torch.nn.functional as F
from transformers import AutoProcessor, ClapModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. CLAP 기반 유사도 함수 - 오디오-캡션 간 유사도 계산

processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)

# 입력: audio_waveform (NumPy 형태 오디오 배열), 프롬포트(캡션) / 출력: [0, 1] 범위의 Float 타입 유사도 
def compute_clap_similarity(audio_waveform: np.ndarray, text: str) -> float:
    inputs = processor(audios=audio_waveform, return_tensors="pt", sampling_rate=48000)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 이미 학습되어 생성된 오디오-텍스트와 비교 => gradient 계산이 불필요(=파라미터 업데이트 X)
    with torch.no_grad():
        audio_embed = clap_model.get_audio_features(**inputs) # 오디오 입력에 대한 CLAP 오디오 임베딩 추출
        # CLAP의 텍스트 인코더를 통해 텍스트 임베딩 추출
        text_embed = clap_model.get_text_features(**processor(text=text, return_tensors="pt", padding=True).to(device))
        # 임베딩을 L2 정규화 => 두 값의 내적 == 코사인 유사도
        audio_embed = F.normalize(audio_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        similarity = (audio_embed @ text_embed.T).item()
        return (similarity + 1) / 2 # [-1,1] -> [0,1] 정규화


# 2. 데이터셋 클래스
class MusicCapsDataset(Dataset):

# 데이터셋 초기화
    '''  
    1.csv_path: 메타데이터 CSV 파일 경로  
    2.audio_dir: 오디오 파일 폴더 경로
    3.max_samples: 학습에 사용할 최대 샘플 수
    4.target_duration: 사용할 오디오 길이 (초)
    5.sampling_rate: 오디오 재샘플링 주파수 (기본 48kHz)   
    
    '''

    def __init__(self, csv_path, audio_dir, max_samples=100, target_duration=10, sampling_rate=48000):
        self.data = pd.read_csv(csv_path).head(max_samples) # pandas 데이터 프레임 구성 - max_samples만큼만 불러오기
        self.audio_dir = audio_dir
        self.target_duration = target_duration # 타겟 오디오 길이 == 10초
        self.sampling_rate = sampling_rate
        self.target_length = target_duration * sampling_rate # 오디오의 길이를 고정 / 10초 × 48,000Hz = 480,000 샘플
        self.data = self.data[self.data["caption"].notna()] # 캡션이 없는 샘플 제거

    def __len__(self):
        return len(self.data) # 데이터셋 전체 개수

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        ytid = row["ytid"] # 오디오 파일 이름 - 유튜브 ID
        caption = row["caption"] # 오디오에 대한 캡션
        start_s = row.get("start_s", 0) # 시작 시간(초) - 없을 경우 0으로 설정

        audio_path = os.path.join(self.audio_dir, f"{ytid}.wav")
        audio, sr = torchaudio.load(audio_path)
        audio = audio[0] # mono - 다채널 오디오의 경우 첫 번째 채널(보통 왼쪽 채널)만 선택하여 1D로 변환

        # 샘플링 레이트 통일 - 48kHz로 resample
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            audio = resampler(audio)
        
        # 샘플 길이 - end_sample - start_sample
        start_sample = int(start_s * self.sampling_rate)
        end_sample = start_sample + self.target_length

        # Zero-padding - 오디오 길이가 target_length보다 짧을 경우
        if end_sample > audio.shape[0]:
            audio = torch.nn.functional.pad(audio, (0, end_sample - audio.shape[0]))

        audio = audio[start_sample:end_sample]
        return audio, caption # 오디오, 캡션 반환


# 3. 실험 실행 함수
def run_experiment(config):
    # 프로젝트 이름, 실험 이름, 설정, 그룹명을 지정하여 wandb에 기록
    # group=config["mode"] 으로 여러 실험을 그룹핑하여 비교
    wandb.init(project="AudioLDM-with-LoRA", name=config["name"], config=config, group=config["mode"])
    accelerator = Accelerator(
        gradient_accumulation_steps=1, # 매 step마다 바로 optimizer.step() 실행 => 바로 가중치 업데이트
        mixed_precision=None, # 모든 연산을 float32(기본 정밀도)로 처리, 안정적 학습 그러나 속도 느림, 메모리 사용량 많음
        log_with="wandb",
        project_config=ProjectConfiguration(project_dir="./", logging_dir="./wandb_logs")
    )

    # 현재 accelerator 상태를 콘솔에 출력 (메인 프로세스만)
    logger = get_logger(__name__)
    logger.info(accelerator.state, main_process_only=True)

    # AudioLDM 전체 파이프라인 불러오기
    pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2")
    text_encoder = pipe.text_encoder # 텍스트 인코더만 분리 => LoRA를 적용할 변수로 지정

    # LoRA 설정값: rank, alpha, 타켓 모듈 - 이 config가 반영된 Adapter 레이어 삽입입
    lora_config = LoraConfig(
        r=config["rank"], # LoRA의 랭크 차원 수
        lora_alpha=config["alpha"], # 스케일 계수, 파라미터 업데이트 폭을 조절하는 정규화 기능(lora_alpha / r)
        inference_mode=False, # 학습 모드로 설정, LoRA 레이어의 파라미터가 requires_grad=True => 옵티마이저로 업데이트 가능
        init_lora_weights="gaussian", # LoRA weight 초기화 방식 설정, 현재 정규분포로 설정
        target_modules=config["target_modules"] # LoRA를 적용할 모델 내 파라미터
    )

    # LoRA 설정을 기반으로 text_encoder에 adapter 레이어 삽입(text_encoder만 requries_grad=True)
    text_encoder = get_peft_model(text_encoder, lora_config)
    
    # text_encoder,VAE와 UNet은 freeze, get_peft_model() 호출 이후 삽입된 LoRA adapter의 파라미터만 학습
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    pipe.text_encoder = text_encoder.to(accelerator.device)
    pipe.to(accelerator.device)

    # 학습 가능한 파라미터만 필터링(requires_grad)하여 AdamW로 학습 진행
    optimizer = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, text_encoder.parameters()),
        lr=config["learning_rate"]
    )

    # 데이터셋 로드 - 전처리된 오디오-캡션 로드
    csv_path = "./data/musiccaps/musiccaps-hiphop_real.csv"
    audio_dir = "./data/musiccaps/audio"
    dataset = MusicCapsDataset(csv_path, audio_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # 배치 크기 1로 하나씩 처리, shuffle=False로 순서 고정정

    clap_scores, losses = [], []
    audio_index = 1

    for audios, prompts in dataloader:
        prompt = prompts[0]
        ref_audio = audios[0].float().to(accelerator.device) # 기준 오디오를 float32로 변환

        # 하나의 오디오에 대해 epochs_per_audio만큼 반복 학습 
        for epoch in range(config["epochs_per_audio"]):
            optimizer.zero_grad() # 기존 gradient 초기화
            # AudioLDM으로 4초 길이의 오디오 생성
            # num_inference_steps: 오디오 생성 시 사용하는 Diffusion 복원 과정의 스텝 수(denosing)
            result = pipe(prompt, num_inference_steps=30, audio_length_in_s=4.0) 
            gen_audio = result["audios"][0]

            # numpy 오디오를 PyTorch 텐서로 변환 / loss 계산을 위한 gradient 활성화 => loss.backward()에서 업데이트 가능하게
            gen_tensor = torch.tensor(gen_audio, dtype=torch.float32, requires_grad=True).to(accelerator.device)
            if gen_tensor.shape[0] > ref_audio.shape[0]:
                gen_tensor = gen_tensor[:ref_audio.shape[0]]
            else:
                gen_tensor = F.pad(gen_tensor, (0, ref_audio.shape[0] - gen_tensor.shape[0]))
            
            # 생성 오디오와 기준 오디오 간의 MSE 손실 계산 후 학습
            loss = F.mse_loss(gen_tensor, ref_audio)
            accelerator.backward(loss) # BackPropagation
            optimizer.step() # Weight 업데이트
            clap_score = compute_clap_similarity(gen_audio, prompt) # 오디오 - 캡션 간 CLAP 유사도 계산

            wandb.log({
                f"Audio {audio_index} | loss": loss.item(),
                f"Audio {audio_index} | clap_score": clap_score,
                f"Audio {audio_index} | audio": wandb.Audio(gen_audio, sample_rate=16000, caption=prompt)
            })
            clap_scores.append(clap_score)
            losses.append(loss.item())
        audio_index += 1

    # 학습된 LoRA weight 저장
    os.makedirs("./data/LoRA_weight", exist_ok=True)
    text_encoder.save_pretrained(f"./data/LoRA_weight/{config['name']}")

    wandb.log({
    "average_clap_score": np.mean(clap_scores),
    "average_loss": np.mean(losses)
})
    # 최종 결과를 wandb summary에 저장 => 전체 실험 비교
    wandb.summary["final_clap_score"] = float(np.mean(clap_scores))
    wandb.summary["final_loss"] = float(np.mean(losses))
    wandb.finish()

    # 실험 결과 JSON 형식으로 저장
    os.makedirs("./results", exist_ok=True)
    with open(f"./results/result_{config['name']}.json", "w") as f:
        json.dump({
            "experiment": config["name"],
            "targets": config["target_modules"],
            "rank": config["rank"],
            "alpha": config["alpha"],
            "clap_score": float(np.mean(clap_scores)),
            "loss": float(np.mean(losses))
        }, f, indent=4)



# 4. 반복 실험 - 4개의 RANK와 3개의 WEIGHT 조합 == 총 12개 실험
if __name__ == "__main__":
    torch.manual_seed(42)

    ranks = [1, 2, 4, 8]
    
    # 실험할 LoRA weight 
    experiment_sets = {
        "Wq": [f"text_model.encoder.layer.{i}.attention.self.query" for i in range(12)], # Wq만
        "Wv": [f"text_model.encoder.layer.{i}.attention.self.value" for i in range(12)], # Wv만
        "WqWv": [f"text_model.encoder.layer.{i}.attention.self.query" for i in range(12)] +
                [f"text_model.encoder.layer.{i}.attention.self.value" for i in range(12)], # Wq, Wv에 모두 적용
    }
    # 각 실험의 파라미터들로 rank, target 모듈, 이름이 바뀜
    # 각 오디오당 5 epoch 반복
    # mode, alpha, learning_rate는 고정
    for rank in ranks:
        for weight, target_modules in experiment_sets.items():
            name = f"LoRA_{weight}_rank{rank}"
            config = {
                "mode": "text_encoder",
                "target_modules": target_modules,
                "rank": rank,
                "alpha": 8,
                "learning_rate": 1e-4,
                "name": name,
                "epochs_per_audio": 5
            }
            run_experiment(config)
