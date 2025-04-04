import torch
from diffusers import AudioLDMPipeline

base_model_id = "cvssp/audioldm-s-full-v2"
model = AudioLDMPipeline.from_pretrained(base_model_id)
print(model.dtype)  # 현재 모델의 데이터 타입 출력
