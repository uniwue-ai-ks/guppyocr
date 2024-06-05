from dataclasses import asdict, dataclass
import enum
import functools
import itertools
import os
from pathlib import Path
from glob import glob
import random
import re
from typing import Iterable, List, Optional, Set, Tuple
import unicodedata
from libpagexml3.libpagexml3 import PageXMLParser, PageXMLParserMode, TextLine, TextEquiv, PageXMLTextTypes
from loguru import logger
import numpy as np
import pandas as pd
from sqlalchemy import all_
from sympy import O
from torch import Value
from tqdm import tqdm
from guppyocr.make_ocr_ds import PolyExtractionSettings, get_color_companion, get_textline_polygon_bb
from docopt import docopt

class LineType(enum.Enum):
    LATIN = 1
    GREEK = 2
    MIXED = 3
    SKIPPED = 4
    PENDING = 5
    REMOVED = 6
    GREEK_IN_LATIN_GT_REMOVED = 7


@dataclass
class GTLine:
    line_id: int
    xml_file: str
    xml_line_id: str
    line_type: LineType
    latin: str
    greek: str
    poly: Tuple[int]
    region_type: str
    
    gt_type_dual: Optional[str] = None
    gt_type_greek: Optional[str] = None
    gt_type_cls: Optional[str] = None
    gt_type_latin: Optional[str] = None

    grp_count: Optional[int] = None



def save_cutout(l: GTLine, o_name: str):
    ofile = Path(o_name).with_suffix(".png")
    comp = get_color_companion_cached(l.xml_file)
    image = comp.crop(l.poly)
    image.save(ofile, compression=1)



@functools.lru_cache(maxsize=2)
def load_pxml(p: Path):
    return PageXMLParser.from_file(p,strictness=PageXMLParserMode.EASY).get_page()

@functools.lru_cache(maxsize=1)
def get_color_companion_cached(p: str):
    return get_color_companion(Path(p))


valid_greek_old: Set[str] = {'ἶ', 'Κ', 'ἇ', 'γ', 'ξ', 'α', 'ἴ', 'ὴ', 'ἄ', 'ὶ', 'ἔ', 'ἥ', 'ἱ', 'ύ', 'ὠ', 'ʼ', 'ά', 'ἷ', 'Β', 'ή', 'Ξ', 'Γ', 'ἠ', 'μ', 'Ρ', 'υ', 'ώ', 'ὥ', 'ψ', 'Τ', 'ἦ', 'Χ', ' ', 'ᾳ', 'φ', 'ᾦ', 'Ν', 'ἑ', 'χ', ':', 'ὔ', 'ε', 'Ο', 'ϊ', 'θ', 'β', 'ὐ', 'ι', 'ἵ', 'ὃ', 'ὂ', '́', 'δ', 'Λ', 'ῷ', 'ΐ', 'ῆ', 'Υ', 'ἃ', 'ὰ', 'ρ', 'ἣ', '´', 'ί', 'ἐ', 'ᾒ', 'ὅ', 'ἓ', 'έ', '?', 'έ', 'ώ', 'ζ', 'ῇ', 'ᾴ', 'κ', 'Η', ',', 'ῥ', '.', ';', 'ἀ', 'π', 'ἰ', 'ὡ', 'ὺ', '΄', 'ᾀ', 'ύ', '·', 'ν', 'ὖ', 'Ζ', '-', 'ἁ', 'ὁ', 'ἧ', 'ῶ', '(', 'Μ', '*', 'ὓ', 'Α', 'ὑ', 'ό', 'σ', 'ό', '↵', 'ὗ', 'ὕ', 'λ', 'Ι', 'ὤ', 'ὣ', 'ὢ', "'", 'ὼ', 'ἆ', 'Ἀ', '"', 'ῃ', '·', 'ῖ', 'ὸ', 'ὒ', 'ῦ', 'ω', '᾿', 'Σ', 'ᾶ', 'ῄ', 'ς', 'η', 'Δ', 'ὄ', 'ᾧ', '&', 'ο', 'ἤ', 'ἕ', ')', 'ῤ', 'ἢ', 'ᾖ', 'ἂ', 'ἒ', 'ᾗ', 'ί', 'ὧ', 'Ω', 'ᾷ', 'ὀ', 'ὲ', 'ἅ', 'ϋ', 'Θ', 'ή', 'ὦ', 'ά', 'Ε', 'ᾄ', 'τ', 'ἳ', 'ῴ', 'Φ', 'ῳ', 'ἡ', 'Π'}
valid_greek: Set[str] = {'Ὕ', 'Ρ', 'ε', 'ἱ', 'ὀ', 'Ὀ', 'ὸ', 'ὁ', 'Ζ', 'ᾀ', 'μ', 'ἣ', '΄', 'ὦ', 'ἇ', '·', 'Π', 'ύ', 'ν', 'ὣ', 'ὒ', 'ὓ', 'ʼ', 'ὂ', 'ᾠ', 'Ἀ', 'ὺ', 'χ', 'ὥ', 'ἡ', 'ὗ', 'ᾼ', 'ἓ', 'ἵ', 'ὖ', 'ἑ', 'ᾗ', 'ί', 'ὐ', 'ὢ', 'ᾴ', '(', 'ᾅ', 'ἰ', 'ᾄ', 'ἒ', 'ᾒ', '⸗', 'ὧ', 'ό', 'γ', "'", 'ύ', 'ᾷ', 'ῆ', '†', 'έ', ']', 'ῦ', 'ὡ', 'Ν', 'ᾔ', 'Ὡ', 'ῷ', 'ὕ', 'ἧ', ':', 'ῥ', 'ή', 'Σ', 'ἕ', ')', 'Τ', '&', '.', 'ζ', 'κ', 'ἆ', 'ὶ', 'ϋ', 'ἂ', '*', 'ἶ', 'Φ', 'Ἡ', 'ᾤ', 'φ', '-', 'Η', 'Ὠ', 'ὔ', 'ή', 'Ἐ', 'ο', 'ῃ', '´', ' ', 'ἦ', 'ὃ', 'η', 'ἤ', 'Ε', 'Α', 'ὴ', 'τ', '?', 'Ἁ', 'ἢ', 'ὠ', 'ψ', 'Υ', 'Μ', 'Ἓ', 'ᾦ', 'ῇ', 'Ὄ', 'ώ', 'ῶ', 'ς', 'ᾐ', '́', 'ά', 'Ψ', 'υ', ';', 'ῳ', 'Ω', 'έ', 'ἔ', 'ϊ', 'Χ', 'ἁ', 'ῂ', 'α', 'ΰ', 'Β', 'Ύ', 'Ῥ', 'Ο', 'ἃ', 'ρ', 'ὅ', 'ώ', '῀', 'ὤ', '᾿', 'Δ', 'Γ', 'Κ', 'λ', 'ἲ', 'ό', 'Ἄ', 'Ξ', 'Λ', 'Ἰ', '`', 'ὼ', 'ῴ', 'ῤ', 'ω', 'ἴ', ',', 'Θ', 'Ὁ', 'δ', 'Ὑ', 'ᾖ', 'ἄ', 'ἷ', 'ῖ', 'ί', 'ὑ', '"', 'Ι', 'π', '↵', 'Ἥ', 'Ἔ', 'ἐ', 'ἥ', 'θ', 'ΐ', 'Ἢ', 'ἀ', 'ᾶ', 'ά', '·', 'ἅ', 'σ', 'ᾂ', 'ῄ', 'ὲ', 'Ἠ', 'ᾳ', 'ὰ', 'β', 'ξ', 'ᾧ', 'ἳ', 'ὄ', 'ἠ', 'ι'}
symbols = {' ', '"', '&', "'", '(', ')', '[', ']', '*', ',', '-', '.', ':', ';', '?'}
numeric = {str(x) for x in range(0,10)}
all_valid_greek = Set.union(valid_greek, symbols, numeric)
def set_line_type_for_line(l: GTLine):
    if l.line_type == LineType.SKIPPED or l.line_type == LineType.REMOVED:
        return
    if l.line_type != LineType.PENDING:
        logger.error(f"Invalid line Type before setting line types for line {l}")
        return
    if l.greek and not l.latin:
        logger.error(f"{l} has greek transcription but no latin GT, skipping")
        l.line_type = LineType.REMOVED
        return
    
    
    if l.greek is None and l.latin is not None:
        if "@" in l.latin:
            l.line_type = LineType.REMOVED
            return
        else:
            l.line_type = LineType.LATIN
            return
    if l.latin is None:
        l.line_type = LineType.REMOVED
        return
    
    line_chars = set(l.greek)
    latin_chars = set(l.latin)

    for lc in latin_chars:
        if lc in valid_greek and lc != ' ' and lc not in symbols and lc not in numeric:
            logger.warning(f"Skipping: {l.latin} because '{lc}' is Greek in Latin GT")
            l.line_type = LineType.GREEK_IN_LATIN_GT_REMOVED
            return

    greek_specific = line_chars - latin_chars
    do_skip = False
    for c in sorted(greek_specific):
        unicode_name = unicodedata.name(c)
        if "LATIN" in unicode_name:
            logger.warning(f"Skipping: {l.greek} [{unicode_name}] ({l.latin})")
            do_skip = True

        if "DIGIT" in unicode_name:
            logger.warning(f"Skipping: {l.greek} [{unicode_name}] ({l.latin})")
            do_skip = True
        
        if c not in all_valid_greek:
            logger.warning(f"Invalid Greek char {c} in {l.greek} in line {l.line_id}")
            do_skip = True
        
        if do_skip:
            l.line_type = LineType.REMOVED
            return
    has_latin_letters = False
    for c in latin_chars:
        try:
            uniname = unicodedata.name(c)
            if "LATIN" in uniname and "LETTER" in uniname:
                has_latin_letters = True
                break
        except ValueError:
            # TODO: fix invalid chars
            pass
    
    if not has_latin_letters:
        l.line_type = LineType.GREEK
        return
    else:
        l.line_type = LineType.MIXED
        return
        

poly_extraction_settings = PolyExtractionSettings(use_region_polys=True)
def make_line_table(files: List[Path]):
    line_id_counter = itertools.count(1)
    all_lines = []

    for f in tqdm(files, desc="Fetching Lines"):
        pxml = load_pxml(f)
        
        for tl in pxml.text_lines():
            parent_region = pxml.region_by_id(tl.parent_region_id)
            if not parent_region: continue
            if not parent_region.region_text_type: continue
            gtl = GTLine(
                line_id=next(line_id_counter),
                xml_file = str(f),
                xml_line_id=tl.id,
                line_type=LineType.SKIPPED,
                latin=tl.text_equivs[TextEquiv.TEXT_ID_GROUND_TRUTH].text if TextEquiv.TEXT_ID_GROUND_TRUTH in tl.text_equivs else None,
                greek=tl.text_equivs[TextEquiv.TEXT_ID_GREEK_CORRECTION].text if TextEquiv.TEXT_ID_GREEK_CORRECTION in tl.text_equivs else None,
                poly=get_textline_polygon_bb(parent_region.polygon),
                region_type=parent_region.region_text_type.value
            )
            all_lines.append(gtl)

            if not gtl.latin:
                continue

            parent_region = pxml.region_by_id(tl.parent_region_id)
            if parent_region.region_text_type in {PageXMLTextTypes.DROP_CAPITAL, PageXMLTextTypes.CAPITAL_WORD}:
                continue

            gtl.line_type = LineType.PENDING
    return all_lines

def build_line_table(globs: List[str], out_dir: str):

    files = sum([glob(x) for x in globs],start=[])
    files = sorted(files)

    all_lines = make_line_table(files)
    fix_line_endings(all_lines)
    
    for l in tqdm(all_lines,desc="Setting line types"):
        set_line_type_for_line(l)
        
    build_gt_types(all_lines)
    df = pd.DataFrame([asdict(x) for x in all_lines])

    with pd.ExcelWriter(Path(out_dir) / "ds5_out.xlsx",engine="openpyxl") as ew:
        df.to_excel(ew)
    return all_lines
    
source_dirs = [
    "/home/norbert/uni/page-segment/GreekPaperNew/Source_Train/oc0149/*.xml",
    "/home/norbert/uni/page-segment/GreekPaperNew/Source_Train/oc0625/*.xml",
    "/home/norbert/uni/page-segment/GreekPaperNew/Source_Train/oc0879_lv/*.xml",
    "/home/norbert/uni/page-segment/GreekPaperNew/Source_Train/oc1095/*.xml"
]

def fix_line_endings(lt: List[GTLine]):
    for l in lt:
        if l.latin:
            lbpos = l.latin.find("↵")
            if lbpos != -1:
                l.latin = l.latin[:lbpos]
        if l.greek:
            lbpos = l.greek.find("↵")
            if lbpos != -1:
                l.greek = l.greek[:lbpos]

multiple_space_replace_regex = re.compile(r"\s\s+")
multiple_repl_regex = re.compile(r"[@][@]+")

def build_gt_types(lt: List[GTLine]):
    
    for l in lt:
        if l.line_type in {LineType.LATIN}:
            # BUILD Latin GT
            l.gt_type_latin = l.latin.strip()
            l.gt_type_latin = multiple_space_replace_regex.sub(" ",l.gt_type_latin)

        if l.line_type in {LineType.LATIN, LineType.GREEK, LineType.MIXED}:
            # BUILD DUAL_LANG_GT
            source = l.greek if l.greek is not None else l.latin
            l.gt_type_dual = source.strip()
            l.gt_type_dual = multiple_space_replace_regex.sub(" ",l.gt_type_dual)

        if l.line_type in {LineType.LATIN, LineType.GREEK, LineType.MIXED}:
            # BUILD CLS_LANG_GT
            l.gt_type_cls = l.latin.strip() # TODO: optionally shrink
            l.gt_type_cls = multiple_space_replace_regex.sub(" ", l.gt_type_cls)

        if l.line_type in {LineType.GREEK, LineType.MIXED}:
            # BUILD GT FOR GREEK
            l.gt_type_greek = "".join( (x if x in valid_greek else "@") for x in l.greek)
            # strip the text
            l.gt_type_greek = l.gt_type_greek.strip()
            l.gt_type_greek = multiple_repl_regex.sub("@", l.gt_type_greek)
            l.gt_type_greek = multiple_space_replace_regex.sub(" ", l.gt_type_greek)



def build_training_ds_n(line_table: List[GTLine], out_dir: str):
    rng = random.Random(123456)
    latin = [x for x in line_table if x.line_type == LineType.LATIN]
    rng.shuffle(latin)
    mixed = [x for x in line_table if x.line_type == LineType.MIXED] + [x for x in line_table if x.line_type == LineType.GREEK]
    rng.shuffle(mixed)



    for c in [latin,mixed]:
        for i,x  in enumerate(c,start=0):
            x.grp_count = i
    
    counts = [500, 1000, 2000, 4000, 100000]
    for count in counts:
        ds_latin = sorted(latin[:count],key=lambda x: x.xml_file)
        ds_mixed = sorted(mixed[:count], key=lambda x: x.xml_file)
        
        out_dir = Path(out_dir)

        #o_dir = f"/home/norbert/uni/page-segment/GreekPaperNew/DS4/C{count}"
        o_dir = Path(out_dir/f"C{count}")

        os.makedirs(o_dir, exist_ok=True)
        os.makedirs(Path(o_dir) / "Latin", exist_ok=True)
        os.makedirs(Path(o_dir) / "CLS", exist_ok=True)
        os.makedirs(Path(o_dir) / "Greek", exist_ok=True)
        os.makedirs(Path(o_dir) / "Dual", exist_ok=True)

        # make the pure latin ds
        for x in tqdm(ds_latin, desc=f"Latin {count}"):
            o_name = Path(o_dir) / "Latin" / f"{x.line_id}_{Path(x.xml_file).name}.txt"
            with open( o_name, "w") as f:
                f.write(x.gt_type_latin)
            save_cutout(x,o_name)
        
        for x in tqdm(ds_mixed + ds_latin, desc=f"CLS {count}"):
            o_name = Path(o_dir) / "CLS" / f"{x.line_id}_{Path(x.xml_file).name}.txt"
            with open( o_name, "w") as f:
                f.write(x.gt_type_cls)
            save_cutout(x,o_name)

        for x in tqdm(ds_mixed, desc=f"Greek {count}"):
            o_name = Path(o_dir) / "Greek" / f"{x.line_id}_{Path(x.xml_file).name}.txt"
            with open( o_name, "w") as f:
                f.write(x.gt_type_greek)
            save_cutout(x,o_name)

        for x in tqdm(ds_latin + ds_mixed, desc=f"Dual {count}"):
            o_name = Path(o_dir) / "Dual" / f"{x.line_id}_{Path(x.xml_file).name}.txt"
            with open( o_name, "w") as f:
                f.write(x.gt_type_dual)
            save_cutout(x,o_name)



USAGE="""make_paper_ds.py

Usage:
    make_paper_ds <out_dir> <datasets>...

"""



if __name__=="__main__":
    args = docopt(USAGE)
    source_dirs = args["<datasets>"]
    line_table = build_line_table(source_dirs, args["<out_dir>"])

    for v in LineType:
        print(f"COUNT({v}): {sum(1 for x in line_table if x.line_type == v)}")
    valid_line_types = {LineType.GREEK, LineType.MIXED, LineType.LATIN}
    filtered_line_table = [x for x in line_table if x.line_type in valid_line_types]

    build_training_ds_n(filtered_line_table, args["<out_dir>"])