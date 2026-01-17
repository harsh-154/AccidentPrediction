import os
import shutil
import pandas as pd

raw_dir = "dataset/raw_videos"
label_file = "dataset/labels.csv"
output_dir = "dataset/separated"

df = pd.read_csv(label_file)

os.makedirs(f"{output_dir}/normal", exist_ok=True)
os.makedirs(f"{output_dir}/accident", exist_ok=True)

for _, row in df.iterrows():
    src = os.path.join(raw_dir, row['filename'])
    if row['label'] == "normal":
        shutil.copy(src, f"{output_dir}/normal")
    else:
        shutil.copy(src, f"{output_dir}/accident")

print("✅ Videos separated into normal and accident.")
