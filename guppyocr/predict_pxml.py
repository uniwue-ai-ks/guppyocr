import glob
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import docopt
import editdistance
import numpy as np
import torch
from libpagexml3.libpagexml3 import PageXMLParser, PageXMLParserMode, TextEquiv, PageXMLBuilder, PageXMLPage
from tqdm import tqdm
from guppyocr.dataset.dataset import resize_with_pad
from guppyocr.make_ocr_ds import extract_pagexml, PolyExtractionSettings, CutoutSample
from guppyocr.train_calamares import ModelParameters, ModelConfiguration, decode_ctc
from typing import Tuple

USAGE="""predict

Usage:
    predict_pxml.py [-d] [--output_dir=DIR] [--glob] --model=MODEL <pagexml>...

Options:
    -d                  Dry run. Do not save
    --output_dir=DIR    Save to specific dir
    --glob              Glob input files
"""

def preprocess_image(img: np.ndarray, width, height) -> Tuple[torch.Tensor, Tuple[float,float]]:
    IMG = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    IMG, ratio_deltaw = resize_with_pad(IMG,(width,height),(255,255,255))
    #IMG = cv2.resize(IMG, (width, height)) # resize
    IMG = (IMG - 127.5)/127.5 # normalization to [-1,1]
    IMG = torch.FloatTensor(IMG) # convert to tensor [H, W, C]
    IMG = IMG.permute(2,0,1) # [C, H, W]
    return IMG, ratio_deltaw

def preprocess_cutout(co: CutoutSample, width, height) -> Tuple[torch.Tensor, Tuple[float,float]]:
    from guppyocr.dataset.dataset import resize_with_pad
    IMG = co.img.convert("RGB")
    IMG = np.array(IMG)
    return preprocess_image(IMG,width,height)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class GuppyPageXMLPredictor:
    def __init__(self, model_path: Path, dry_run: bool = False) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dry_run = dry_run
        self.model_config, self.model = ModelConfiguration.load_model(model_path, self.device)
        self.model.eval()
        #print("Warning, Greek Transcription is enabled")
    def predict_pagexml(self, f: Path) -> PageXMLPage:
        with torch.set_grad_enabled(False):
            page = PageXMLParser.from_file(f,strictness=PageXMLParserMode.EASY).get_page()    
            #cutouts = list(extract_pagexml(page,Path(f),PolyExtractionSettings(text_id=TextEquiv.TEXT_ID_GREEK_CORRECTION,fallback_text_id=TextEquiv.TEXT_ID_GROUND_TRUTH),inference=True))
            cutouts = list(extract_pagexml(page,Path(f),PolyExtractionSettings(),inference=True))
            # TODO: deactivate this again
            #for i, cutout in enumerate(cutouts):
            #    cutout.img.save(f"/tmp/cutouts/{Path(f).with_suffix('').name}_{i:03}.png")
            imgs = [preprocess_cutout(x,self.model_config.mconfig.Width, self.model_config.mconfig.Height)[0] for x in cutouts]
        
            for img, co in zip(imgs, cutouts):
                img = img[None, :, : ,:]
                img = img.to(self.device)
                prediction, _ , _ ,_ = self.model.forward(img,None)
                decoded_tgt = decode_ctc(prediction[0], self.model_config.mconfig.id2char)
                co.tl.text_equivs[TextEquiv.TEXT_ID_PREDICTION] = TextEquiv(TextEquiv.TEXT_ID_PREDICTION,decoded_tgt)

            # build new pagexml
            if not self.dry_run:
                build_pxml = PageXMLBuilder.build_pagexml(page,"GuppyOcr")
                with open(f,"w") as of:
                    of.write(build_pxml)
            return page

def main():
    args = docopt.docopt(USAGE)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args["-d"]:
        dry_run = True
        print("Dry run, saving disabled")
    else:
        dry_run = False
    mc, model = ModelConfiguration.load_model(args["--model"], device)
    model = model.eval()
    pagexmls = args["<pagexml>"]
    if len(pagexmls) == 1 and args["--glob"]:
        pagexmls = sorted(glob.glob(pagexmls[0]))
    total_chars = 0
    total_errors = 0
    with torch.no_grad():
        for f in tqdm(pagexmls):
            page = PageXMLParser.from_file(f,strictness=PageXMLParserMode.EASY).get_page()
            cutouts = list(extract_pagexml(page,Path(f),PolyExtractionSettings(use_region_polys=True, text_id=TextEquiv.TEXT_ID_GREEK_CORRECTION,fallback_text_id=TextEquiv.TEXT_ID_GROUND_TRUTH),inference=True))

            #for i, cutout in enumerate(cutouts):
            #    cutout.img.save(f"/tmp/cutouts/{Path(f).with_suffix('').name}_{i:03}.png")

            imgs = [preprocess_cutout(x,mc.mconfig.Width, mc.mconfig.Height)[0] for x in cutouts]

            had_gt = False
            for orig_img, co in zip(imgs, cutouts):
                img = orig_img[None, :, : ,:]

                img = img.to(device)
                prediction, _ , _ ,_ = model.forward(img,None)
                decoded_tgt = decode_ctc(prediction[0], mc.mconfig.id2char)
                
                co.tl.text_equivs[TextEquiv.TEXT_ID_PREDICTION] = TextEquiv(TextEquiv.TEXT_ID_PREDICTION,decoded_tgt)
                """if co.tl.has_gt_text():
                    gt_text = co.tl.get_text_gt().replace("↵","")
                    distance = editdistance.distance(decoded_tgt, gt_text)
                    total_chars += len(gt_text)
                    total_errors += distance
                    print("GT:  ", gt_text)
                    print("Pred:", decoded_tgt)
                    #print("Pred:", decoded_tgt)
                    #print("GT:  ", gt_text)
                    #print("CER: ", editdistance.distance(decoded_tgt, gt_text) / len(gt_text))
                    had_gt = True
                """
                if (co.tl.has_gt_text() and TextEquiv.TEXT_ID_GREEK_CORRECTION in co.tl.text_equivs) or \
                   (co.tl.has_gt_text() and TextEquiv.TEXT_ID_GREEK_CORRECTION not in co.tl.text_equivs and "@" not in co.tl.get_text_gt()):

                    groundtruth_text = co.gt_text
                    groundtruth_text = groundtruth_text.replace("↵","")
                    distance = editdistance.distance(decoded_tgt, groundtruth_text)
                    import PIL.Image
                    #if groundtruth_text == "mat, κεῖνός τε κεῖνα στεργέτω κἀγὼ τάδε.":
                    #    Image.fromarray(co.).save("/tmp/gt.png")
                    if len(groundtruth_text) != 0:
                        total_chars += len(groundtruth_text)
                        total_errors += distance
                    print("GT:  ", groundtruth_text)
                    print("Pred:", decoded_tgt)
                    print("Distance: ", distance, "CER: ", distance / max(1,len(groundtruth_text)))

                
                """
                if TextEquiv.TEXT_ID_GREEK_CORRECTION in co.tl.text_equivs:
                    gt_text = co.tl.text_equivs[TextEquiv.TEXT_ID_GREEK_CORRECTION].text.replace("↵","") # TODO: this should be removed for the final version
                    distance = editdistance.distance(decoded_tgt, gt_text)
                    total_chars += len(gt_text)
                    total_errors += distance
                    print("GT:  ", gt_text)
                    print("Pred:", decoded_tgt)
                    #print("Pred:", decoded_tgt)
                    #print("GT:  ", gt_text)
                    #print("CER: ", editdistance.distance(decoded_tgt, gt_text) / len(gt_text))
                    had_gt = True
                """
            
            #if dry_run and not had_gt:
            #    print(f"Stopping early at {f} because no GT has been found")
            #    break

            # build new pagexml
            if not dry_run:
                build_pxml = PageXMLBuilder.build_pagexml(page,f"GuppyOCR ({mc.mconfig.arch})")
                output_fname = f
                if args["--output_dir"]:
                    output_fname = Path(args["--output_dir"]) / Path(f).name
                with open(output_fname,"w") as of:
                    of.write(build_pxml)
        if total_chars > 0:
            print(f"Total Errors: {total_errors} / Total Chars. {total_chars}")
            print(f"Estimated CER: {total_errors / total_chars * 100:.4f} %")
        else:
            print(f"Estimated CER: unknown (no GT found)")


if __name__ == "__main__":
    main()
  







