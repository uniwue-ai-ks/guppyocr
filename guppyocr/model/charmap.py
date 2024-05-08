import itertools
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

from tqdm import tqdm

from guppyocr.ocr_dataset import IDatasetReader


@dataclass
class CharMap:
    charmap: List[str]
    char2id: Dict[str, int]
    id2char: Dict[int, str]

def load_charmap(cmap: Path):
    with open(cmap) as f:
        mp = json.loads(f.read())
        charmap = mp["charmap"]
        id2char = dict(zip(itertools.count(), charmap))
        char2id = dict(zip(charmap, itertools.count()))
        return CharMap(charmap,char2id=char2id,id2char=id2char)

def save_charmap(out_path: Path, cmap: CharMap):
    json_content = {"charmap": cmap}

    import json
    with open(out_path, "w") as f:
        f.write((json.dumps(json_content)))

def make_charmap_from_readers(readers: List[IDatasetReader], use_tqdm=False) -> Tuple[CharMap, int]:
    all_chars = set()
    max_seq_len = 0
    for reader in readers:
        if use_tqdm:
            iterator = tqdm(range(len(reader)),desc="Building Charmap")
        else:
            iterator = range(len(reader))
        for i in iterator:
            txt = reader.get_text(i).replace("\n","")
            if len(txt) > max_seq_len:
                max_seq_len = len(txt)
            if len(txt) > 1000:
                print(f"ERROR: {txt}")
            for c in txt:
                all_chars.add(c)

    charmap = ["PAD"] + sorted(list(all_chars)) + ["UNK", "END"]
    id2char = dict(zip(itertools.count(), charmap))
    char2id = dict(zip(charmap, itertools.count()))
    return CharMap(charmap,char2id=char2id,id2char=id2char), max_seq_len

def make_charmap_from_input_dirs(dirs: List[str]) -> CharMap:
    all_chars = set()
    for dir in dirs:
        for x in Path(dir).iterdir():
            if x.is_file() and x.suffixes == [".gt",".txt"]:
                with open(x) as file:
                    s = file.read()
                    for c in s.replace("\n",""):
                        all_chars.add(c)

    charmap = ["PAD"] + sorted(list(all_chars)) + ["UNK", "END"]
    id2char = dict(zip(itertools.count(), charmap))
    char2id = dict(zip(charmap, itertools.count()))
    return CharMap(charmap,char2id=char2id,id2char=id2char)

if __name__ == "__main__":
    ap = ArgumentParser("make_charmap")
    ap.add_argument("--src",type=str,required=True)
    ap.add_argument("--out", type=str, required=True)

    args = ap.parse_args()

    charmap = make_charmap_from_input_dirs([args.src]).charmap

    save_charmap(args.out, charmap)

