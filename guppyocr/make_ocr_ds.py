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
    PageRegionTypes, TextEquiv
import PIL
import PIL.Image
import PIL.ImageDraw
from tqdm import tqdm
from guppyocr.ocr_dataset import IDatasetWriter, PlainDatasetWriter, make_dataset_writer



USAGE="""Make OCR DS

Usage:
    make_ocr_ds [options] <output_path> <pagexmls>...

Options:
    --writer=WRITER         Output writer to use [default: plain]
    --prefix=PREFIX         set image prefix [default: ]
    --skip                  Skip files if no Companion file is found
    --no-greek              Skip lines containing greek characters (@)
    --squash_ats            Squash At Characters into a single char
    --glob                  Glob Image Files instead of iterating
    --text_equiv_id=TEXT_ID
"""

class FilenameEnumerator:
    def __init__(self, start_count: int, suffix: str):
        self.count = itertools.count(start_count)
        self.suffix = suffix

    def next(self, prefix: str = None):
        x = next(self.count)
        return f"{prefix}_{x:07}{self.suffix}"

class CompanionImageType(enum.Enum):
    COLOR = "color"
    BINARIZED = "binarized"


class CompanionFileNotFound(Exception):
    pass

@dataclass
class PolyExtractionSettings:
    mask_poly: bool = False
    use_companion_image: CompanionImageType = CompanionImageType.COLOR
    use_region_polys: bool = True
    text_id: int = TextEquiv.TEXT_ID_GROUND_TRUTH
    fallback_text_id: int = None

@dataclass
class CutoutSample:
    img: PIL.Image.Image
    gt_text: str
    tl: TextLine


def extract_textline_image(img: np.ndarray, poly: np.ndarray, extraction_settings: PolyExtractionSettings) -> PIL.Image.Image:
    #assert len(img.shape) == 3, f"Image is not in XYC format, it is {img.shape}"

    # get the bb of the polygon
    x_min, x_max = floor(np.amin(poly[:,0])), ceil(np.amax(poly[:,0]))
    y_min, y_max = floor(np.amin(poly[:,1])), ceil(np.amax(poly[:,1]))

    img_w, img_h = x_max - x_min + 1, y_max - y_min + 1

    if extraction_settings.mask_poly:
        raise NotImplementedError("Mask Poly is currently not implemented currently")
        # make a new image to put the polygon into
        #poly_img = PIL.Image.new("1",(img_w,img_h),0)
        #poly_draw = PIL.ImageDraw.ImageDraw(poly_img)
        #poly_draw.polygon(poly.tolist(),1,0,0)
        #show_images([np.array(poly_img)], "PolyImg")

    cut_image = PIL.Image.fromarray(img[y_min:y_max+1,x_min:x_max+1])
    return cut_image

@dataclass
class RegionFilter:
    region_type_blacklist: Set[PageRegionTypes] = field(default_factory=set)
    text_type_blacklist: Set[PageXMLTextTypes] = field(default_factory=set)


def extract_pagexml(pxml: PageXMLPage, xml_path: Path, poly_extraction_settings: PolyExtractionSettings, inference: bool = False, filter: RegionFilter = RegionFilter()) -> Iterable[CutoutSample]:
    # load the companion image
    if poly_extraction_settings.use_companion_image == CompanionImageType.COLOR:
        for cn in [xml_path.with_suffix(".png"), xml_path.with_suffix(".jpg"), xml_path.with_suffix(".jpeg"),xml_path.with_suffix(".png.ocr") ]:
            if cn.exists():
                companion_image = PIL.Image.open(cn)
                companion_image = companion_image.convert("RGB")
                break
        else:
            raise CompanionFileNotFound(f"Companion image for {xml_path} not found")
    else:
        companion_image = PIL.Image.open(xml_path.with_suffix(".png.ocr"))

    companion_image_arr = np.array(companion_image)

    for tl in pxml.text_lines():
        if not inference and not poly_extraction_settings.text_id in tl.text_equivs and not poly_extraction_settings.fallback_text_id in tl.text_equivs:
            continue
        parent_region = pxml.region_by_id(tl.parent_region_id)
        if parent_region.region_type in filter.region_type_blacklist or \
           parent_region.region_text_type in filter.text_type_blacklist:
            continue

        if poly_extraction_settings.use_region_polys:
            poly = parent_region.polygon
        else:
            poly = tl.polygon
        gt_txt = ""
        if poly_extraction_settings.text_id in tl.text_equivs:
            gt_txt = tl.text_equivs[poly_extraction_settings.text_id].text
        elif poly_extraction_settings.fallback_text_id in tl.text_equivs:
            gt_txt = tl.text_equivs[poly_extraction_settings.fallback_text_id].text
        cut_image = extract_textline_image(companion_image_arr,poly,poly_extraction_settings)
        yield CutoutSample(cut_image,gt_txt, tl)

def find_all_pagexmls_recursively(folder: Path) -> List[Path]:
    pagexmls = []
    for x in folder.iterdir():
        if x.is_dir():
            pagexmls += find_all_pagexmls_recursively(x)
        elif x.suffix == ".xml":
            pagexmls.append(x)
    return pagexmls

@dataclass
class LanguageSettings:
    disable_greek: bool = False
    cut_line_ending_mark: bool = True
    filter_small_lines: int = 0
    gt_id:int = TextEquiv.TEXT_ID_GREEK_CORRECTION
    gt_fallback: int = TextEquiv.TEXT_ID_GROUND_TRUTH

def make_dataset(dataset_writer: IDatasetWriter, companion_image_type: CompanionImageType, source_files_dirs: List[Path], glob_input: bool = False, language_settings: LanguageSettings = LanguageSettings(), ds_prefix: str = ''):
    settings = PolyExtractionSettings(
        mask_poly=False,
        use_companion_image=CompanionImageType.COLOR,
        use_region_polys=False,
        text_id=language_settings.gt_id,
        fallback_text_id=language_settings.gt_fallback
    )
    
    if glob_input:
        xml_files = list(itertools.chain.from_iterable([sorted(glob.glob(path)) for path in source_files_dirs]))
    else:
        xml_files = []
        for arg in [Path(x) for x in source_files_dirs]:
            if arg.is_dir():
                xml_files += find_all_pagexmls_recursively(arg)
            elif arg.suffix == ".xml":
                xml_files.append(arg)
    
    for x in tqdm(xml_files):
        if ds_prefix:
            prefix = "_".join([ds_prefix, Path(x).with_suffix("").name])
        else:
            prefix = Path(x).with_suffix("").name
        fn = FilenameEnumerator(0,"")
        try:
            page = PageXMLParser.from_file(x,strictness=PageXMLParserMode.EASY).get_page()
            for cutout in extract_pagexml(page,Path(x),settings):
                if  language_settings.disable_greek and "@" in cutout.gt_text:
                    continue

                if len(cutout.gt_text) < language_settings.filter_small_lines:
                    continue
                if language_settings.cut_line_ending_mark:
                    lbpos = cutout.gt_text.find("↵")
                    if lbpos != -1:
                        cutout.gt_text = cutout.gt_text[:lbpos]
        
                try:
                    dataset_writer.write(str(fn.next(prefix)), cutout.img , cutout.gt_text)
                except FileExistsError as e:
                    print("Skipping:" + e.strerror)

        except CompanionFileNotFound as e:
            print(f"Skipping {x} (no companion file)")
    dataset_writer.close()


def main(args):
    output_path = Path(args["<output_path>"])
    
    writer: IDatasetWriter = make_dataset_writer(output_path, args["--writer"])

    make_dataset(writer, 
        CompanionImageType.COLOR, 
        args["<pagexmls>"],args["--glob"],
        LanguageSettings(disable_greek=args["--no-greek"],
                         cut_line_ending_mark=True,
                         filter_small_lines=3,
                         gt_id=TextEquiv.TEXT_ID_GREEK_CORRECTION,
                         gt_fallback=TextEquiv.TEXT_ID_GROUND_TRUTH
                         )
        )


if __name__ == "__main__":
    args = docopt(USAGE)
    main(args)
