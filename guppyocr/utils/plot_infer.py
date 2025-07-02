from pathlib import Path
from guppyocr.api import GuppyOCR
import PIL
import numpy as np
from matplotlib import pyplot as plt
import scipy

def plot_infer(img: Path, guppyocr: GuppyOCR):
    co = np.array(PIL.Image.open(img).convert("RGB"))
    outp, scale_factor = guppyocr.ocr_with_char_position(co)
    p_map = scipy.special.softmax(outp.char_mapping.probability_map, axis=1)
    # squash empty columns
    cols = np.sum(p_map,axis=0) > 0.1
    p_map = np.transpose(p_map[:,cols])
    plt.imshow(p_map, cmap="hot", interpolation='nearest')
    #outp.char_mapping.probability_map
    #plt.imshow(co)
    plt.show()
    print(outp.decoded_string)



if __name__ == "__main__":
    # load the model
    
    ocr = GuppyOCR.load_model("/home/norbert/uni/page-segment/guppy-models/Complete2A1.pth")

    img_path = Path("/home/norbert/uni/studiolo2024/imgreek/source.png")
    plot_infer(img_path, ocr)