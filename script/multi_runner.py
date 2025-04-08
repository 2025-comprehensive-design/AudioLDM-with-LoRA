import itertools
import subprocess
import os
import json
from datetime import datetime

# 실험 파라미터 조합 정의
target_combinations = [
    ["to_q"], ["to_v"], ["to_q", "to_v"], ["to_q", "to_k"],
    ["to_q", "to_k", "to_v"], ["to_q", "to_v", "to_out"]
]
ranks = [4, 8, 16]
alphas = [4, 8, 16]

# 실험 스크립트 경로
train_script = "train_audioldm_lora.py"
results_log = "lora_experiments_results.json"

# 결과 저장 딕셔너리
experiment_results = []

# 결과를 txt로도 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
text_log_file = f"lora_results_{timestamp}.txt"

# 실험 반복 실행
for targets, rank, alpha in itertools.product(target_combinations, ranks, alphas):
    # 실험 이름
    experiment_name = f"lora_{'-'.join(targets)}_r{rank}_a{alpha}"

    # 환경변수로 전달할 실험 구성
    os.environ["LORA_TARGET_MODULES"] = ",".join(targets)
    os.environ["LORA_RANK"] = str(rank)
    os.environ["LORA_ALPHA"] = str(alpha)
    os.environ["WANDB_NAME"] = experiment_name  # wandb run name
    os.environ["WANDB_TAGS"] = f"{'-'.join(targets)},rank{rank},alpha{alpha}"

    # 학습 실행
    print(f"\n[실험 시작] {experiment_name}")
    try:
        subprocess.run(["python", train_script], check=True)
        print(f"[완료] {experiment_name}")

        result = {
            "experiment": experiment_name,
            "targets": targets,
            "rank": rank,
            "alpha": alpha,
            "clap_score": None,
            "loss": None
        }
        experiment_results.append(result)

        # 텍스트 로그 저장
        with open(text_log_file, "a") as log_f:
            log_f.write(f"[완료] {experiment_name}\n")

    except subprocess.CalledProcessError:
        print(f"[오류 발생] 실험 실패: {experiment_name}")
        with open(text_log_file, "a") as log_f:
            log_f.write(f"[실패] {experiment_name}\n")

# 전체 결과 JSON 저장
with open(results_log, "w") as f:
    json.dump(experiment_results, f, indent=4)

print("\n모든 실험 종료. 결과가 저장되었습니다 ->", results_log, "및", text_log_file)
