from typing import Union

import torch

from abc import ABC, abstractmethod
from os import PathLike
import numpy.typing

from guppyocr.model.model import ModelConfiguration
from guppyocr.train_calamares import decode_ctc
from guppyocr.predict_pxml import preprocess_image


class OCREngineBase(ABC):
    @abstractmethod
    def ocr(self, cutout: numpy.typing.ArrayLike) -> str:
        pass

class InvalidInputImage(Exception):
    pass
class GuppyOCR(OCREngineBase):
    def __init__(self, mc: ModelConfiguration, model: torch.nn.Module, device: torch.device):
        self.mc = mc
        self.model = model
        self.device = device


    @staticmethod
    def load_model(file: Union[str,PathLike], device: torch.device = None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        mc, model = ModelConfiguration.load_model(file,device)
        return GuppyOCR(mc, model, device)


    def ocr(self, cutout: numpy.typing.ArrayLike) -> str:
        if not len(cutout.shape) == 3:
            raise InvalidInputImage("Shape of Image cutout must be 3")

        img = preprocess_image(cutout, self.mc.mconfig.Width, self.mc.mconfig.Height)
        img = img[None, :, : ,:]
        with torch.no_grad():
            img = img.to(self.device)
            prediction, _ , _ ,_ = self.model.forward(img,None)
            decoded_tgt = decode_ctc(prediction[0], self.mc.mconfig.id2char)
        return decoded_tgt

