from guppyocr.model.network import CalamariNetv2, CRNN, CRNNTiny, STNCRNN, TrCRNN
from dataclasses import dataclass
import dataclasses
from typing import Union, Dict, Optional
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
        elif mconfig.arch == "trcrnn":
            return TrCRNN(mconfig.Channel,mconfig.Height,mconfig.Width,mconfig.seq_len,len(mconfig.voc),mconfig.hidden_units,mconfig.layers,mconfig.keep_prob,mconfig.seq_len,device,mconfig.leaky_relu)
        else:
            raise KeyError(f"invalid model specified: {mconfig.arch}")

    @staticmethod
    def load_model(path: Union[str, os.PathLike, BinaryIO, IO[bytes]], device) -> Tuple['ModelConfiguration', nn.Module]:
        ndict = torch.load(path, map_location=lambda storage, loc: storage)
        mconfig = ModelParameters(**ndict["mconfig"])
        config = ModelConfiguration(mconfig)
        model = config.make_model(device)
        model.load_state_dict(ndict["model_state"], strict=False)
        model = model.to(device)
        #model = torch.nn.DataParallel(model).to(device)
        return config, model



@dataclass
class AdjustVocResult:
    voc: List[str]
    char2id: Dict[str,int]
    id2char: Dict[int,str]

    mconfig: ModelConfiguration
    model: nn.Module
    new_chars: Optional[List] = None



def adjust_voc_model(new_voc, mconfig: ModelConfiguration, model, device) -> Tuple[bool, AdjustVocResult]:
    if mconfig.mconfig.arch != "crnn":
        raise NotImplementedError()
    old_voc_set = set(mconfig.mconfig.voc)
    new_voc_set = set(new_voc)
    if new_voc_set.issubset(old_voc_set):
        return False, AdjustVocResult(
            voc=mconfig.mconfig.voc,
            char2id=mconfig.mconfig.char2id,
            id2char=mconfig.mconfig.id2char,
            mconfig=mconfig,
            model=model)
    else:
        # make a new voc, char2id and id2char
        difference = list(new_voc_set.difference(old_voc_set))
        print("Charset in model differ. New chars to add: ", difference)
        new_voc = mconfig.mconfig.voc + difference
        char2id = {x:i for i,x in enumerate(new_voc)}
        id2char = {i:x for i,x in enumerate(new_voc)}
        mconfig.mconfig.voc = new_voc
        mconfig.mconfig.id2char = id2char
        mconfig.mconfig.char2id = char2id
        model: CRNN
        model.output_classes = len(new_voc)
        old_layer = model.rnn[-1].embedding
        new_layer = nn.Linear(old_layer.in_features,model.output_classes)
        model.rnn[-1].embedding = new_layer
        model = model.to(device)
        print(f"Replaced output layer in model from {old_layer.out_features} to {new_layer.out_features} characters")



        return True, AdjustVocResult(
            voc=new_voc,
            char2id=char2id,
            id2char=id2char,
            mconfig=mconfig,
            model=model,
            new_chars=list(difference)
        )
