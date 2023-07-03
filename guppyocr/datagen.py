import itertools
import random
import sys
from pathlib import Path
from typing import List, Tuple
import PIL.Image
from tqdm import tqdm
import stellaris

from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)
from trdg.string_generator import create_strings_from_dict


# The generators use the same arguments as the CLI, only as parameters
def wordlist():
    with open("/home/norbert/uni/page-segment/sar-pytorch/wordlist") as f:
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

words = wordlist()

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
for i in range(5000):
    sent = " ".join([random.choice(words) for _ in range(random.randint(1,3))])
    wl = random.choice(words)
    wl = wl[:random.randint(3,len(wl))]
    sent += " " + wl + "."
    train_data.append(sent)
"""
#sentences += [stupid_gen(words) for _ in range(3000)]

random.shuffle(train_data)


# add DIW data
diw_lines = []
print("Sentences without DIW strings: " + str(len(sentences)))
"""  a
with open("diw-texts.txt") as f:
    lines = f.readlines()
    random.shuffle(lines)
    sentences += lines[:4000]
"""
random.shuffle(sentences)

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
    ])
])

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

random.seed(12345)
data_strings = []
for i in tqdm(range(1000000)):
    data_strings.append(RandomStringGenerator(None))


fonts=[str(x) for x in Path("/home/norbert/uni/page-segment/textgenerator/fonts/googlefonts/").iterdir()]

random.shuffle(fonts)
fonts_iter = itertools.cycle(fonts)


for i, text in tqdm(enumerate(data_strings), total=len(data_strings)):    # Do something with the pillow images here.
    font = next(fonts_iter)
    gen = GeneratorFromStrings(
        [text],
        blur=OneOf([
            (1,lambda x: 0),
            (1, lambda x: 1),
            ])(None),
        size=48,
        skewing_angle=OneOf([
            (1, lambda x: 1),
            (1, lambda x: 0)
            ])(None),
        random_skew=True,
        fonts=[font],
        margins=OneOf([
            (1, lambda x: (5, 5, 5, 5)),
            (1, lambda x: (3, 3, 3, 3)),
            (1, lambda x: (1, 1, 1, 1)),
        ])(None),
    )
    try:
        img, _ = gen.next()
        if img == None:
            continue
        fname = Path("/tmp/googlefontsrng/")/ f"{i:07}.png"
        gt_path = Path(fname).with_suffix(".gt.txt")
        with open(gt_path,"w") as f:
            f.write(text)

        img: PIL.Image.Image = img
        img.save(fname,compression=1)
        if i == len(data_strings):
            break
    except Exception as e:
        print(f"Error genrating with font: {Path(font).name}")
        print(e)


if __name__ == "__main__":
    pass