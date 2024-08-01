from pathlib import Path
import shutil
from tqdm import tqdm

SOURCE_PATH="/home/norbert/uni/page-segment/greekocr/ByWriters2/Oporinus"

OUTPUT_DIR="/home/norbert/uni/page-segment/greekocr/ByWriters2/Split/Oporinus"

train_dir = Path(OUTPUT_DIR) / "train"
test_dir = Path(OUTPUT_DIR) / "test"
import os

os.makedirs(train_dir,exist_ok=False)
os.makedirs(test_dir,exist_ok=False)

import glob

files = sorted(glob.glob(SOURCE_PATH + "/*.txt"))

split_idx = round(len(files) * 0.8)

train_split = files[:split_idx]
test_split = files[split_idx:]

for f in tqdm(train_split, desc="Train Split"):
    shutil.copyfile(f,train_dir / Path(f).name)
    comp = Path(f).with_suffix(".png")
    shutil.copyfile(comp,train_dir / Path(comp).name)


for f in tqdm(test_split, desc="Test split"):
    shutil.copyfile(f,test_dir / Path(f).name)
    comp = Path(f).with_suffix(".png")
    shutil.copyfile(comp,test_dir / Path(comp).name)


