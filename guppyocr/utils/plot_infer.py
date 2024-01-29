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
    plt.imshow(p_map, cmap="hot", interpolation='nearest')
    #outp.char_mapping.probability_map
    #plt.imshow(co)
    plt.show()




if __name__ == "__main__":
    # load the model
    ocr = GuppyOCR.load_model("models/Complete1A4.pth")

    img_path = Path("/home/norbert/uni/studiolo2024/ctc_in.png")
    plot_infer(img_path, ocr)