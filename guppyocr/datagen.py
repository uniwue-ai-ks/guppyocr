import itertools
import multiprocessing
import random
import sys
from pathlib import Path
from typing import List, Tuple
import PIL.Image
from tqdm import tqdm
#import stellaris

from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)
from trdg.string_generator import create_strings_from_dict


# The generators use the same arguments as the CLI, only as parameters
def wordlist(path: str):
    with open(path) as f:
        lines = [x for x in f.readlines()]
    return lines


def numbers_gen():
    if random.random() > 0.5:
        return str(random.randint(0,10000))
    elif random.random() < 0.6:
        return str(random.randint(0,1000)) + "," + str(random.randint(0,99))
    elif random.random() < 0.5:
        return "€" + str(random.randint(0,1000)) + "," + str(random.randint(0,99))
    else:
        return str(random.randint(0,1000)) + "%"



def stupid_gen(words):
    stupid = "#-*/%."
    sentence = ""
    for i in range(random.randint(1,4)):
        if random.random() < 0.5:
            sentence += "".join([random.choice(stupid) for _ in range(random.randint(1,8))])
        if random.random() < 0.5:
            sentence += " "
        if random.random() < 0.5:
            sentence += random.choice(words)
        else:
            sentence += random.choice(words).upper()
    return sentence

"""
sentences = []
for x in create_strings_from_dict(4,True,40000,words):
    if random.random() < 0.7:
        sentences.append(x)
    else:
        sentences.append(x.upper())


numbers = [numbers_gen() for x in range(1000)]

train_data = sentences

for i in range(6000):
    train_data.append(random.choice(sentences) + " " + random.choice(numbers))
    train_data.append(random.choice(numbers) + " " + random.choice(sentences))
"""
"""
for i in range(5000):
    sent = " ".join([random.choice(words) for _ in range(random.randint(1,3))])
    wl = random.choice(words)
    wl = wl[:random.randint(3,len(wl))]
    sent += " " + wl + "."
    train_data.append(sent)
"""
#sentences += [stupid_gen(words) for _ in range(3000)]
"""
random.shuffle(train_data)


# add DIW data
diw_lines = []
"""
#print("Sentences without DIW strings: " + str(len(sentences)))
"""  a
with open("diw-texts.txt") as f:
    lines = f.readlines()
    random.shuffle(lines)
    sentences += lines[:4000]
"""
#random.shuffle(sentences)

def Sequential(fns: List[callable]):
    def new_fn(x):
        for f in fns:
            x = f(x)
        return x
    return new_fn

def OneOf(fns: Tuple[float, callable]):
    sum_weights =  sum([x[0] for x in fns])
    def new_fn(x):
        rng = random.random()
        slim = 0
        for lim, f in fns:
            if rng * sum_weights < lim + slim:
                return f(x)
            slim += lim
        else:
            return fns[-1][1](x)
    return new_fn

def Repeat(min: int, max: int, fn: callable):
    def new_fn(x):
        rng = random.randint(min,max)
        for t in range(rng):
            x = fn(x)
        return x
    return new_fn

def Maybe(prob: float, fn: callable):
    def new_fn(x):
        rng = random.random()
        if rng < prob:
            return fn(x)
        else:
            return x
    return new_fn

RandomNumberGen = Sequential([
        lambda x: str(random.randint(1,10000)),
        OneOf([
            (1, lambda x: x + "," + str(random.randint(1,9999))),
            (1, lambda x: x + "." + str(random.randint(1,9999))),
            (1, lambda x: x),
            (1, lambda x: x + f",{random.randint(1,99):02d}"),
            (1, lambda x: x + f".{random.randint(1,99):02d}")
        ])
    ])

from abc import ABC, abstractmethod

OUTPUT_DIR="/tmp/german_train1/"

def mp_generate(arg):
    text, font, i = arg
    leftm, rightm = random.randint(3,22), random.randint(3,22)
    gen = GeneratorFromStrings(
        [text],
        size=random.choice([28,32,38,42,48]),
        blur=OneOf([
            (1,lambda x: 0),
            (1, lambda x: 0),
            ])(None),
        #size=48,
        skewing_angle=OneOf([
            (1, lambda x: 1),
            (1, lambda x: 0)
            ])(None),
        random_skew=True,
        fonts=[font],
        background_type=OneOf([(3,lambda x: 0),(1,lambda x: 2)])(None),
        margins=OneOf([ # Top, left, bottom, right
            (1, lambda x: (5, leftm, 5, rightm)),
            (1, lambda x: (3, leftm, 3, rightm)),
            (1, lambda x: (3, leftm, 3, rightm)),
            (1, lambda x: (1, rightm, 1, rightm)),
        ])(None),
    )
    try:
        img, _ = gen.next()
        if img == None:
            raise RuntimeError("no image was generated")
        fname = Path(OUTPUT_DIR)/ f"{i:07}.png"
        gt_path = fname.with_suffix(".gt.txt")
        with open(gt_path,"w") as f:
            f.write(text)

        img: PIL.Image.Image = img
        img.save(fname,compression=1)
    except Exception as e:
        print(f"Error genrating with font: {Path(font).name}")
        print(e)
    
def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=False)

    words = wordlist("/home/norbert/uni/page-segment/sar-pytorch/wordlist")
    german_words = wordlist("/home/norbert/uni/page-segment/sar-pytorch/german_wordlist.txt")

    RandomStringGenerator = \
    Sequential([
        lambda x: [],
        Repeat(1,5, OneOf([
            (1, lambda x: x + [RandomNumberGen([])]),
            (1, lambda x: x + [random.choice(words)]),
            (0.2, lambda x: x + ["-"]),
            (0.2, lambda x: x + [","])
        ])),
        lambda x: [y.replace("\n","") for y in x],
        lambda x: " ".join(x),
        lambda x: x.replace(" - ",""),
        lambda x: x.replace(" ,", ","),
    ])

    RandomWordGenerator = \
        OneOf([
            (1, lambda x: random.choice(german_words)),
            (1, lambda x: RandomNumberGen("")),
            (1, lambda x: f"{random.choice(german_words).strip()}-{random.choice(german_words).strip()}"),
            (1, lambda x: random.choice(words))
        ])

    random.seed(123456)
    data_strings = []
    for i in tqdm(range(100000)):
        #data_strings.append(RandomStringGenerator(None))
        data_strings.append(RandomWordGenerator(None))


    #fonts=[str(x) for x in Path("/home/norbert/uni/page-segment/sar-pytorch/fonts/").iterdir()]
    fonts=[str(x) for x in Path("/home/norbert/uni/page-segment/GoogleFonts/fontsv1/").iterdir()]

    random.shuffle(fonts)
    fonts_iter = itertools.cycle(fonts)

    data = [x for x in zip(data_strings, fonts_iter, itertools.count())]
    mp = multiprocessing.Pool()
    for _ in tqdm(mp.imap(mp_generate,data), total=len(data)):
        pass
       


def convert_to_doctr_json(dir: Path):
    import shutil
    def slurp(f: Path):
        with open(f) as inp:
            return inp.read().strip()
    
    images_dir = dir / "images"
    images_dir.mkdir(exist_ok=True)
    labels = {}
    for f in dir.iterdir():
        if f.suffix == ".png":
            shutil.move(str(f),str(images_dir))
            labels[f.name] = slurp(f.with_suffix(".txt"))
    import json
    with open(dir / "labels.json", "w") as json_f:
        json.dump(labels,json_f)
    
if __name__ == "__main__":
    convert_to_doctr_json(Path("/home/norbert/uni/page-segment/Datensets/funsd/training_words"))
    #main()f