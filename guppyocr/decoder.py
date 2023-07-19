from dataclasses import dataclass
from itertools import groupby
from typing import List

import numpy as np


@dataclass
class CharLocationInfo:
    char_start: int
    char_end: int
    char: str


@dataclass
class ExtendedDecoderInfo:
    chars: str
    white_space_index: int
    blank_index: int
    charLocationInfo: List[CharLocationInfo]
    probability_map: np.array = None


@dataclass
class DecoderOutput:
    decoded_string: str
    char_mapping: ExtendedDecoderInfo = None


class GreedyDecoder:
    def __init__(self, chars):
        self.chars = chars

    def decode(self, mat, extended_info: bool = False, remove_whitespace: bool = False) -> DecoderOutput:
        index_list = np.argmax(np.squeeze(mat), axis=1)
        blank_index = 0  # len(self.chars)
        best_chars_collapsed = [self.chars[k] for k, _ in groupby(index_list) if k != blank_index]
        pred_string = ''.join(best_chars_collapsed)
        extendend_info_stats = None
        if extended_info:
            listextf = []

            start, end = 0, -1
            
            try:
                whitespace_char_index = self.chars.index(" ")
            except ValueError:
                whitespace_char_index = None
                remove_whitespace = False
            
            # prev = None
            for x, y in groupby(index_list):
                y_length = len(list(y))
                # prev = x if prev is None else prev
                start = end + 1
                end = end + y_length
                listextf.append(CharLocationInfo(start, end, x))

            listextf = [x for x in listextf if x.char != blank_index]

            if remove_whitespace:
                listextf = [x for x in listextf if x.char != whitespace_char_index]
            

            extendend_info_stats = ExtendedDecoderInfo(chars=self.chars,
                                                       white_space_index=whitespace_char_index,
                                                       blank_index=blank_index,
                                                       charLocationInfo=listextf,
                                                       probability_map=mat
                                                       )
            assert len(listextf) == len(pred_string)
            

        return DecoderOutput(decoded_string=pred_string, char_mapping=extendend_info_stats)
