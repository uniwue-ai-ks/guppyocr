from argparse import ArgumentParser
from collections import defaultdict
from glob import glob
from pathlib import Path
import random
from PIL import Image
from guppyocr.ocr_dataset import ParquetDatasetWriter
import os
import shutil

parse = ArgumentParser("Make Crossfold DS")
parse.add_argument("--input_dir",type=str,required=True)
parse.add_argument("--output_dir",type=str,required=True)
parse.add_argument("--rand", type=int, required=False)
parse.add_argument("--format",type=str,default="pq")
parse.add_argument("--folds",type=int,required=True)

args = parse.parse_args()

args.input_dir = str(args.input_dir).removesuffix("/")
files = sorted(glob(args.input_dir + "/*.txt"))

os.makedirs(args.output_dir,exist_ok=True)

if args.rand is not None:
    # shuffle the dataset
    r = random.Random(args.rand)
    r.shuffle(files)

# make n folds
folds = defaultdict(list)
for i, f in enumerate(files):
    folds[i % args.folds].append(f)

# write the train datasets
for test_fold in range(args.folds):
    foldname = f"fold{test_fold}"
    pw = ParquetDatasetWriter(Path(args.output_dir) / f"train_{foldname}.pq")
    for fold in range(args.folds):
        if fold == test_fold: continue
        for f in folds[fold]:
            img = Image.open(Path(f).with_suffix("").with_suffix(".png"))
            pw.write(Path(f).name,img,Path(f).read_text())
    pw.close()
    # make the test dir
    test_dir = Path(args.output_dir) / f"test_{foldname}"
    os.makedirs(test_dir)
    for f in folds[test_fold]:
        shutil.copyfile(f,test_dir / Path(f).name)
        comp = Path(f).with_suffix("").with_suffix(".png")
        shutil.copyfile(comp,test_dir / Path(comp).name)






