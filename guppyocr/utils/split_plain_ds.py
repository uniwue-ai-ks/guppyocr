from pathlib import Path
import shutil
from tqdm import tqdm
from argparse import ArgumentParser

#parse = ArgumentParser("Split Plain DS")
#parse.add_argument("--input_dir",type=str,required=True)

#a#rgs 
SOURCE_PATH="/tmp/guppym/diw-ocr-big"

OUTPUT_DIR="/tmp/guppym"

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


