import enum
import glob
import itertools
from dataclasses import dataclass, field
from math import ceil, floor
from pathlib import Path
from typing import Iterable, Set, Iterator, List

import cv2
#import h5py as h5py
import numpy as np
from docopt import docopt
from libpagexml3.libpagexml3 import PageXMLPage, PageXMLParser, PageXMLParserMode, TextLine, PageXMLTextTypes, \
    PageRegionTypes
import PIL
import PIL.Image
import PIL.ImageDraw
from tqdm import tqdm
from guppyocr.ocr_dataset import make_dataset_writer, make_dataset_reader


USAGE="""Convert OCR Dataset

Usage:
    make_ocr_ds [options] <input_path> <output_path>

Options:
    --reader=READER     Which reader to use [default: plain]
    --writer=WRITER     Output writer to use [default: plain]
"""


def main(args):
    reader = make_dataset_reader(Path(args["<input_path>"]),format=args["--reader"])
    writer = make_dataset_writer(Path(args["<output_path>"]), format=args["--writer"])
    reader_keys = reader.list_files()

    for i, key in tqdm(enumerate(reader_keys), total=len(reader_keys)):
        img, txt = reader.get(i)
        writer.write(key, img, txt)
    writer.close()

if __name__ == "__main__":
    args = docopt(USAGE)
    main(args)
