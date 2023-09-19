from typing import List
from docopt import docopt
from pathlib import Path
import editdistance

from sqlalchemy import Result
from guppyocr.api import GuppyOCR

from guppyocr.make_ocr_ds import CutoutSample, PolyExtractionSettings, extract_pagexml
from libpagexml3.libpagexml3 import PageXMLParser, PageXMLParserMode
from tqdm import tqdm
import numpy as np

USAGE=""" Font-Finder

Usage:
    font_finder.py <models_dir> <pagexml_dir> <output>

"""



from peewee import *




def main(args):
    # make a list with all available models
    models_list = [x for x in Path(args["<models_dir>"]).iterdir() if x.is_file() and x.suffix == ".pth"]
    pagexml_list = [x for x in Path(args["<pagexml_dir>"]).iterdir() if x.is_file() and x.suffix == ".xml"]


    db = SqliteDatabase(args["<output>"])
    

    class ResultEntry(Model):
        model =  TextField()
        pred =  TextField(null=True)
        gt = TextField(default=False)
        cer = FloatField(default=False)

        class Meta:
            database = db
    db.create_tables([ResultEntry])

    # make the evaluation cutout list
    cutouts: List[CutoutSample] = []
    for f in pagexml_list:
        cutouts.extend(
            extract_pagexml(PageXMLParser.from_file(f,strictness=PageXMLParserMode.EASY).get_page(),f,PolyExtractionSettings())
        )
    cutouts = [x for x in cutouts if len(x.gt_text) > 0 ]
    
    # make the output data structure
    failed_models = []
    models_tqdm = tqdm(models_list, "model")
    for m in models_tqdm:
        models_tqdm.set_description(Path(m).name)
        try:
            result_entries: List[ResultEntry] = []
            predictor = GuppyOCR.load_model(m)
            for co in cutouts:
                res = predictor.ocr(np.array(co.img))
                cer = editdistance.distance(res,co.gt_text) / max(1,len(co.gt_text))
                result_entries.append(ResultEntry(model=str(m.name),pred=res,gt=co.gt_text, cer=float(cer)))

            with db.atomic():
                ResultEntry.bulk_create(result_entries)    

            print("success")
            del predictor
        except Exception as e:
            #raise e
            failed_models.append(m)


if __name__ == "__main__":
    args = docopt(USAGE)
    main(args)
