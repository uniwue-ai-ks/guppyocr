import itertools
import os
import random
import shutil
import sys
import tempfile
from typing import List
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from trdg.generators import (
    GeneratorFromStrings,
)

from guppyocr.datagen import OneOf, RandomNumberGen, Repeat, Sequential, wordlist
from guppyocr.ocr_dataset import MemoryDatasetRW

import numpy as np

from guppyocr.train_calamares import TrainingOpts, train_model

FONTS_INPUT_DIR = Path("/home/norbert/uni/page-segment/textgenerator/fonts/googlefonts")

if Path("working_fonts.json").exists():
    working_fonts = pd.read_json("working_fonts.json")
else:
    working_fonts: List[Path] = []

    for i in tqdm([x for x in FONTS_INPUT_DIR.iterdir() if x.is_file()]):
        try:
            gen = GeneratorFromStrings(
                ["abcedfghijklmnopqrstuVWXYZ1231287z3.572"],
                blur=0,
                size=48,
                skewing_angle=1,
                random_skew=True,
                fonts=[str(i.absolute())],
                margins=(3,3,3,3)
            )
            generated = gen.next()
            working_fonts.append(i)
        except Exception as e:
            print(f"Font {i} not working. Skipping")

    working_fonts = [str(x.absolute()) for x in working_fonts]
    working_fonts: pd.DataFrame = pd.DataFrame(data={"font_path": working_fonts})

    working_fonts.to_json("working_fonts.json")

if Path("strings.json").exists():
    strings = pd.read_json("strings.json")
else:
    words = wordlist()

    string_rng = RandomStringGenerator = \
    Sequential([
        lambda x: [],
        Repeat(2,4, OneOf([
            (1, lambda x: x + [RandomNumberGen([])]),
            (3, lambda x: x + [random.choice(words)]),
            (0.2, lambda x: x + ["-"]),
            (0.2, lambda x: (x + [","]) if len(x) > 0 else x)
        ])),
        lambda x: [y.replace("\n","") for y in x],
        lambda x: " ".join(x),
        lambda x: x.replace(" - ",""),
        lambda x: x.replace(" ,", ","),
    ])
    strings = []
    while len(strings) < 1000:
        new_str = string_rng([])
        if len(new_str) == 0:
            continue
        else:
            strings.append(new_str)

    strings = pd.DataFrame(data={"text": strings})
    strings.to_json("strings.json")

#print(strings)
print(f"Max String length: {max(map(len,strings['text']))}")

from peewee import *

db = SqliteDatabase("models.db")

class Trained(Model):
    font_file =  TextField()
    model_file =  TextField(null=True)
    failed = BooleanField(default=False)
    training_started = BooleanField(default=False)

    class Meta:
        database = db

db.create_tables(models=[Trained])

if Trained.select().count() == 0:
    for f in tqdm(working_fonts["font_path"], desc="Filling database with fonts"):
        Trained(font_file=f).save()

text_strings = list(strings["text"])

progress = tqdm(itertools.count())
progress_iter = iter(progress)
while next(progress_iter) or True:
    try:
        todo = Trained.get(Trained.model_file == None, Trained.failed != True, Trained.training_started == False)
    except:
        break
    progress.set_description(f"Train Font: {str(Path(todo.font_file).name)}")
    todo.training_started = True
    todo.save()
        
    try:
        # make an in Memory dataset using this font
        random.seed(0)    
        train_ds = MemoryDatasetRW()
        val_ds = train_ds

        gen = GeneratorFromStrings(
            text_strings,
            blur=2,
            size=48,
            skewing_angle=1,
            random_skew=True,
            fonts=[str(todo.font_file)],
            margins=(3,3,3,3)
        )

        for _ in tqdm(text_strings,desc="Genering Dataset"):
            img, txt = gen.next()
            train_ds.write(txt,np.array(img),txt)
        
        with tempfile.TemporaryDirectory() as d:
            tropts = TrainingOpts(output="tempdir",dataset=train_ds,val_dataset=val_ds,reader="memory",epoch=25,early_stopping=6,augment=True, gpu=True)
            train_result = train_model(tropts)
            os.makedirs("font_models_out",exist_ok=True)
            out_path = (Path("font_models_out") / Path(todo.font_file).name).with_suffix(".pth")
            shutil.copyfile(train_result.best_model_path,out_path)
            todo.model_path = str(out_path.absolute())
            todo.failed = False
            todo.save()
     
    except Exception as e:
        todo.failed = True
        todo.save()

        


