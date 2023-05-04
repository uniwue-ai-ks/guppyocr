import random
import sys
from pathlib import Path

import PIL.Image
from tqdm import tqdm

from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)
from trdg.string_generator import create_strings_from_dict


# The generators use the same arguments as the CLI, only as parameters

def wordlist():
    with open("wordlist") as f:
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

generator = GeneratorFromStrings(
    train_data,
    blur=0,
    random_blur=True,
    size=48,
    skewing_angle=0,

    random_skew=True,

    fonts=[str(x) for x in Path("fonts_simple").iterdir()]
)

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
for i, (img, lbl) in tqdm(enumerate(generator)):    # Do something with the pillow images here.
    fname = Path("simple_data1")/ f"{i:07}.png"
    gt_path = Path(fname).with_suffix(".gt.txt")
    with open(gt_path,"w") as f:
        lbl = lbl.replace('\n', ' ')
        lbl = lbl.replace('  ', ' ')
        f.write(lbl)

    img: PIL.Image.Image = img
    img.save(fname,compression=1)
    if i == 60000:
        break



if __name__ == "__main__":
    pass