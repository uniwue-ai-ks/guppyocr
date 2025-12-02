from pathlib import Path
import shutil
from tqdm import tqdm
from argparse import ArgumentParser
import random

parse = ArgumentParser("Split Plain DS")
parse.add_argument("--input_dir",type=str,required=True)
parse.add_argument("--output_dir",type=str,required=True)
parse.add_argument("--seq", type=bool)
parse.add_argument("--rand", type=int)



args = parse.parse_args()

assert (args.seq is True) ^ (args.rand is not None), "Either seq or Rand must be set"
#a#rgs 
SOURCE_PATH=args.input_dir

OUTPUT_DIR=args.output_dir

train_dir = Path(OUTPUT_DIR) / "train"
test_dir = Path(OUTPUT_DIR) / "test"
import os

os.makedirs(train_dir,exist_ok=False)
os.makedirs(test_dir,exist_ok=False)

import glob

files = sorted(glob.glob(SOURCE_PATH + "/*.txt"))

split_idx = round(len(files) * 0.8)

if args.seq == True:
    train_split = files[:split_idx]
    test_split = files[split_idx:]
elif args.rand is not None:
    r = random.Random(args.rand)
    r.shuffle(files)
    train_split = files[:split_idx]
    test_split = files[split_idx:]
else:
    raise NotImplementedError("Either seq or rand must be set")

for f in tqdm(train_split, desc="Train Split"):
    shutil.copyfile(f,train_dir / Path(f).name)
    comp = Path(f).with_suffix(".png")
    shutil.copyfile(comp,train_dir / Path(comp).name)


for f in tqdm(test_split, desc="Test split"):
    shutil.copyfile(f,test_dir / Path(f).name)
    comp = Path(f).with_suffix(".png")
    shutil.copyfile(comp,test_dir / Path(comp).name)


