# 완성된 모델 확인용 배포 코드 포함해야 한다.
from diffusers import AudioLDMPipeline
import scipy
import torch

repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
# 학습 가중치 불러오기
# lora_weight_pth = ""
# pipe.unet.load_attn_procs(lora_weight_pth)

prompt = "The Hip Hop song features a flat male vocal rapping over repetitive, echoing piano melody, claps and punchy kick hits. At the very beginning of the loop, there are stuttering hi hats, sustained synth bass and filtered, pitched up female chant playing. It sounds groovy and addictive - thanks to the way rappers are rapping. The subgenre of hip-hop is boom bap"
audio = pipe(prompt, num_inference_steps=200, audio_length_in_s=10.0).audios[0]

scipy.io.wavfile.write("Base003.wav", rate=16000, data=audio)
