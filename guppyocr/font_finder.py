from docopt import docopt
from pathlib import Path
from guppyocr.api import GuppyOCR

from guppyocr.make_ocr_ds import extract_pagexml
from libpagexml3.libpagexml3 import PageXMLParser, PageXMLParserMode

USAGE=""" Font-Finder

Usage:
    font_finder.py <models_dir> <pagexml_dir>

"""





def main(args):
    # make a list with all available models
    models_list = [x for x in Path(args["<models_dir>"]).iterdir() if x.is_file() and x.suffix == ".pth"]
    pagexml_list = [x for x in Path(args["<pagexml_dir>"]) if x.is_file() and x.suffix == ".xml"]

    # make the evaluation cutout list
    cutouts = []
    for f in pagexml_list:
        cutouts.extend(
            extract_pagexml(PageXMLParser.from_file(f,strictness=PageXMLParserMode.EASY).get_page(),f)
        )
    
    # make the output data structure
    results = []
    failed_models = []
    for m in models_list:
        try:
            predictor = GuppyOCR.load_model(m)

    


        except Exception as e:
            print(e)
            failed_models.append(m)







if __name__ == "__main__":
    args = docopt(USAGE)
    main(args)
