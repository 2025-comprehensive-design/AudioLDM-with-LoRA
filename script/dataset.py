from datasets import load_dataset

raw = load_dataset("rkstgr/mtg-jamendo", split="train[:200]", trust_remote_code=True)
hiphop_count = 0

for i, sample in enumerate(raw):
    genres = [g.lower() for g in sample["genres"]]
    print(f"[{i}] genres: {sample['genres']}")
    if "hiphop" in genres:
        hiphop_count += 1

print(f"\n총 5000개 중 'hiphop' 장르 샘플 수: {hiphop_count}")