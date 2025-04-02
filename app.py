# 완성된 모델 확인용 배포 코드 포함해야 한다.
'''
from diffusers import AudioLDMPipeline
import scipy
import torch

repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "hiphop music"
audio = pipe(prompt, num_inference_steps=200, audio_length_in_s=10.0).audios[0]

scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)
'''

