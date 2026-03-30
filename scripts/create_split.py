"""
Generate train/val split files (80/20, stratified by village).
Outputs: preprocessed_dataset/train.txt, preprocessed_dataset/val.txt
"""
import os
import random

random.seed(42)

img_dir = "preprocessed_dataset/images"
all_files = sorted(os.listdir(img_dir))

# Group by village
villages = {}
for f in all_files:
    # village_id is everything before the last two _Y_X.tif parts
    parts = f.rsplit("_", 2)
    vid = "_".join(parts[:-2]) if len(parts) >= 3 else f
    villages.setdefault(vid, []).append(f)

train_list = []
val_list = []

for vid, patches in sorted(villages.items()):
    random.shuffle(patches)
    split_idx = int(len(patches) * 0.8)
    train_list.extend(patches[:split_idx])
    val_list.extend(patches[split_idx:])

# Write split files
with open("preprocessed_dataset/train.txt", "w") as f:
    f.write("\n".join(sorted(train_list)) + "\n")

with open("preprocessed_dataset/val.txt", "w") as f:
    f.write("\n".join(sorted(val_list)) + "\n")

print(f"Train: {len(train_list)} patches")
print(f"Val:   {len(val_list)} patches")
print(f"Total: {len(train_list) + len(val_list)}")
print(f"\nPer-village breakdown:")
for vid, patches in sorted(villages.items()):
    n = len(patches)
    t = int(n * 0.8)
    print(f"  {vid}: {t} train / {n - t} val (total {n})")
