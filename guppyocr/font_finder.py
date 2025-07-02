from audioop import avg
from typing import List
from docopt import docopt
from pathlib import Path
import editdistance

from sqlalchemy import Result
from guppyocr.api import GuppyOCR
from guppyocr.decoder import DecoderOutput

from guppyocr.make_ocr_ds import CutoutSample, PolyExtractionSettings, extract_pagexml
from libpagexml3.libpagexml3 import PageXMLParser, PageXMLParserMode
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
import pandas as pd

USAGE=""" Font-Finder

Usage:
    font_finder.py <models_dir> <pagexml_dir> <output>

"""



from peewee import *


def calculate_output_char_confidence(do: DecoderOutput) -> float:
    confidences = []
    probabilities = softmax(do.char_mapping.probability_map,axis=1)
    #pmap = softmax()
    for i, c in enumerate(do.char_mapping.charLocationInfo):
        beg, end = c.char_start, c.char_end + 1
        avg_prob = np.sum(probabilities[beg: end,c.char]) / max(1, end - beg)
        confidences.append(avg_prob)
    if not confidences:
        return 0.0
    return sum(confidences) / max(1,len(confidences))

def main(args):
    # make a list with all available models
    models_list = [x for x in Path(args["<models_dir>"]).iterdir() if x.is_file() and x.suffix == ".pth"]
    pagexml_list = [x for x in Path(args["<pagexml_dir>"]).iterdir() if x.is_file() and x.suffix == ".xml"]


    """db = SqliteDatabase(args["<output>"])
    

    class ResultEntry(Model):
        model =  TextField()
        pred =  TextField(null="")
        gt = TextField(default="")
        cer = FloatField(default=0.0)
        conf = FloatField(default=0.0)

        class Meta:
            database = db
    
    db.create_tables([ResultEntry])
    """
    # make the evaluation cutout list
    cutouts: List[CutoutSample] = []
    for f in pagexml_list:
        raise NotImplementedError() # Next Call is missing an argument (inference)
        #cutouts.extend(
        #    extract_pagexml(PageXMLParser.from_file(f,strictness=PageXMLParserMode.EASY).get_page(),f,!!!!!inference missingPolyExtractionSettings(use_region_polys=False))
        #)
    cutouts = [x for x in cutouts if len(x.gt_text) > 0 ]
    
    # make the output data structure
    failed_models = []
    models_tqdm = tqdm(models_list, "model")
    results = []

    for m in models_tqdm:
        models_tqdm.set_description(Path(m).name)
        confidences = []
        cers = []

        try:
            #result_entries: List[ResultEntry] = []
            predictor = GuppyOCR.load_model(m)
            for co in cutouts:
                #res = predictor.ocr(np.array(co.img))
                
                res2,_ = predictor.ocr_with_char_position(np.array(co.img))
                res = res2.decoded_string
                cer = editdistance.distance(res,co.gt_text) / max(1,len(co.gt_text))
                conf = calculate_output_char_confidence(res2)
                #result_entries.append(ResultEntry(model=str(m.name),pred=res,gt=co.gt_text, cer=float(cer), conf=conf))
                confidences.append(conf)
                cers.append(cer)
            
            """
            with db.atomic():
                ResultEntry.bulk_create(result_entries)    
            """
            results.append((Path(m).name, sum(confidences) / max(1,len(confidences)),sum(cers) / max(1,len(cers))))
            print("success")
            del predictor
        except Exception as e:
            #raise e
            failed_models.append(m)
    df = pd.DataFrame(results,columns=["Model","Conf","CER"])
    df.to_csv(args["<output>"])
if __name__ == "__main__":
    args = docopt(USAGE)
    main(args)
