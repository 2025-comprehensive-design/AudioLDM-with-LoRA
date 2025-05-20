from safetensors.torch import load_file

state_dict = load_file("/home/2020112030/WorkSpace/2025/AudioLDM-with-LoRA/data/LoRA_weight/r2_alpha4/checkpoint-19400/model.safetensors")
print(state_dict.keys())