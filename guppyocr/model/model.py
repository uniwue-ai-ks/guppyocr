from guppyocr.model.network import CalamariNetv2, CRNN, CRNNTiny, STNCRNN
from dataclasses import dataclass
import dataclasses
from typing import Union
from typing import BinaryIO, IO, Tuple, List
import os
import torch
from torch import nn

@dataclass
class ModelParameters:
    Height: int
    Width: int
    Channel: int
    feature_height: int
    feature_width: int
    seq_len: int
    arch: str
    hidden_units: int
    layers: int
    keep_prob: bool
    voc: List[str]
    char2id: dict
    id2char: dict
    leaky_relu: bool = False

class ModelConfiguration:
    def __init__(self, mconfig: ModelParameters):
        self.mconfig = mconfig

    def save(self, path: Union[str, os.PathLike, BinaryIO, IO[bytes]], state_dict: dict):
        ndict = {
            "mconfig": dataclasses.asdict(self.mconfig),
            "model_state": state_dict
        }
        torch.save(ndict, path)
    def make_model(self, device):
        mconfig = self.mconfig
        if mconfig.arch == "calamari":
            return CalamariNetv2(mconfig.Channel,mconfig.Height, mconfig.Width,mconfig.seq_len,len(mconfig.voc),mconfig.hidden_units,mconfig.layers,mconfig.keep_prob,mconfig.seq_len,device,mconfig.leaky_relu)
        elif mconfig.arch == "crnn":
            return CRNN(mconfig.Channel,mconfig.Height,mconfig.Width,mconfig.seq_len,len(mconfig.voc),mconfig.hidden_units,mconfig.layers,mconfig.keep_prob,mconfig.seq_len,device,mconfig.leaky_relu)
        elif mconfig.arch == "stncrnn":
            return STNCRNN(mconfig.Channel,mconfig.Height,mconfig.Width,mconfig.seq_len,len(mconfig.voc),mconfig.hidden_units,mconfig.layers,mconfig.keep_prob,mconfig.seq_len,device,mconfig.leaky_relu)
        else:
            raise KeyError(f"invalid model specified: {mconfig.arch}")

    @staticmethod
    def load_model(path: Union[str, os.PathLike, BinaryIO, IO[bytes]], device) -> Tuple['ModelConfiguration', nn.Module]:
        ndict = torch.load(path, map_location=lambda storage, loc: storage)
        mconfig = ModelParameters(**ndict["mconfig"])
        config = ModelConfiguration(mconfig)
        model = config.make_model(device)
        model.load_state_dict(ndict["model_state"], strict=False)
        model = torch.nn.DataParallel(model).to(device)
        return config, model