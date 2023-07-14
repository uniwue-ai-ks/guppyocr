import string
from pathlib import Path
from typing import Tuple

import cv2
import torch.utils.data as data
import torch
import numpy as np
from guppyocr.ocr_dataset import IDatasetReader


def resize_with_pad(image: np.ndarray,
                    new_shape: Tuple[int, int],
                    padding_color: Tuple[int,int,int] = (255, 255, 255)) -> Tuple[np.array, Tuple[float, float]]:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new/orig for new,orig in zip(new_shape, original_shape)))# float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])


    if new_size[0] > new_shape[0] or new_size[1] > new_shape[1]:
        ratio = float(min(new/orig for new,orig in zip(new_shape, original_shape)))
        new_size = tuple([int(x * ratio) for x in original_shape])

    assert new_size[0] <= new_shape[0] and new_size[1] <= new_shape[1]
    image = cv2.resize(image, new_size)
    #delta_w = new_shape[0] - new_size[0] if new_shape[0] > new_size[0] else 0
    #delta_h = new_shape[1] - new_size[1] if new_shape[1] > new_size[1] else 0
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    #left, right = delta_w//2, delta_w-(delta_w//2)
    left, right = 0, delta_w
    #print(image.shape, top, bottom,left,right)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,None,value=padding_color)
    return image, (ratio, delta_w)

def transform_image(img: np.ndarray, width, height):
    IMG, _ = resize_with_pad(img,(width,height),(255,255,255))
        #IMG = cv2.resize(IMG, (width, height)) # resize
    IMG = (IMG - 127.5)/127.5 # normalization to [-1,1]
    IMG = torch.FloatTensor(IMG) # convert to tensor [H, W, C]
    IMG = IMG.permute(2,0,1) # [C, H, W]
    return IMG

def load_image(filename, width, height):
    IMG = cv2.imread(filename)
    return transform_image(IMG, width, height)

class MySynthDataset(data.Dataset):
    def __init__(self, height, width, seq_len, total_img_path, char2id, putend: bool = False):
        '''
        height: input height to model
        width: input width to model
        total_img_path: path with all images
        annotation_path: mat labeling file
        seq_len: sequence length
        '''
        self.total_img_path = total_img_path
        self.height = height
        self.width = width
        self.seq_len = seq_len
        self.dataset = []
        self.char2id = char2id

        self.char2id = char2id
        self.output_classes = len(self.char2id)
        self.load_images_to_memory = False
        self.putend = putend

        for f in Path(total_img_path).iterdir():
            if f.suffix == ".png":
                # read the text
                with open(f.with_suffix(".gt.txt")) as gtf:
                    gt = gtf.read().replace('\n','')
                    if len(gt) > seq_len: continue
                # load the image
                if self.load_images_to_memory:
                    self.dataset.append((load_image(str(f.absolute()),self.width, self.height), gt))
                else:
                    self.dataset.append((str(f.absolute()), gt))

    def __getitem__(self, index):
        img_name, label = self.dataset[index]
        if self.load_images_to_memory:
            IMG = img_name
        else:
            IMG = load_image(img_name, self.width, self.height)
        y_true = np.ones(self.seq_len,dtype=np.int32)*int(self.char2id['PAD'])
        # label processing
        for i, c in enumerate(label):
            try:
                index = self.char2id[c]
            except KeyError:
                index = self.char2id["UNK"]
            y_true[i] = int(index)
        if self.putend:
            y_true[-1] = self.char2id['END'] # if requested, put END at the back
        y_true = y_true.astype(int)
        # one hot
        y_onehot = np.eye(self.output_classes)[y_true]  # [seq_len, output_classes]

        return IMG, torch.FloatTensor(y_onehot)

    def __len__(self):
        return len(self.dataset)

    def shrink(self, new_len):
        self.dataset = self.dataset[:new_len]
        return self


class MySimpleDataset(data.Dataset):
    def __init__(self, dataset_reader: IDatasetReader, height, width, seq_len, char2id, putend: bool = False):
        '''
        height: input height to model
        width: input width to model
        total_img_path: path with all images
        annotation_path: mat labeling file
        seq_len: sequence length
        '''
        self.reader = dataset_reader
        self.height = height
        self.width = width
        self.seq_len = seq_len
        self.dataset = []
        self.char2id = char2id

        self.char2id = char2id
        self.output_classes = len(self.char2id)
        self.load_images_to_memory = False
        self.putend = putend
        
        for i,f in enumerate(self.reader.list_files()):
                if self.load_images_to_memory:
                    img, gt = self.reader.get(i)
                    img = transform_image(img, width, height)
                    self.dataset.append((img,gt))
                else:
                    gt = self.reader.get_text(i)
                    self.dataset.append((i, gt))

    def __getitem__(self, index):
        img_name, label = self.dataset[index]
        if self.load_images_to_memory:
            IMG = img_name
        else:
            img = self.reader.get_image(img_name)
            IMG = transform_image(img, self.width, self.height)
        
        y_true = np.ones(self.seq_len,dtype=np.int32)*int(self.char2id['PAD'])
        # label processing
        for i, c in enumerate(label):
            try:
                index = self.char2id[c]
            except KeyError:
                index = self.char2id["UNK"]
            y_true[i] = int(index)
        if self.putend:
            y_true[-1] = self.char2id['END'] # if requested, put END at the back
        y_true = y_true.astype(int)
        # one hot
        y_onehot = np.eye(self.output_classes)[y_true]  # [seq_len, output_classes]

        return IMG, torch.FloatTensor(y_onehot)

    def __len__(self):
        return len(self.dataset)

    def shrink(self, new_len):
        self.dataset = self.dataset[:new_len]
        return self

"""
class ParquetDataset:
    def __init__(self, height, width, seq_len, pq_path, char2id, putend: bool = False):
        '''
        height: input height to model
        width: input width to model
        total_img_path: path with all images
        annotation_path: mat labeling file
        seq_len: sequence length
        '''
        self.pq_path = pq_path
        self.height = height
        self.width = width
        self.seq_len = seq_len
        self.dataset = []
        self.char2id = char2id

        self.char2id = char2id
        self.output_classes = len(self.char2id)
        self.load_images_to_memory = False
        self.putend = putend

        for f in Path(total_img_path).iterdir():
            if f.suffix == ".png":
                # read the text
                with open(f.with_suffix(".gt.txt")) as gtf:
                    gt = gtf.read().replace('\n','')
                    if len(gt) > seq_len: continue
                # load the image
                if self.load_images_to_memory:
                    self.dataset.append((load_image(str(f.absolute()),self.width, self.height), gt))
                else:
                    self.dataset.append((str(f.absolute()), gt))

    def __getitem__(self, index):
        img_name, label = self.dataset[index]
        if self.load_images_to_memory:
            IMG = img_name
        else:
            IMG = load_image(img_name, self.width, self.height)
        y_true = np.ones(self.seq_len,dtype=np.int32)*int(self.char2id['PAD'])
        # label processing
        for i, c in enumerate(label):
            try:
                index = self.char2id[c]
            except KeyError:
                index = self.char2id["UNK"]
            y_true[i] = int(index)
        if self.putend:
            y_true[-1] = self.char2id['END'] # if requested, put END at the back
        y_true = y_true.astype(int)
        # one hot
        y_onehot = np.eye(self.output_classes)[y_true]  # [seq_len, output_classes]

        return IMG, torch.FloatTensor(y_onehot)

    def __len__(self):
        return len(self.dataset)

    def shrink(self, new_len):
        self.dataset = self.dataset[:new_len]
        return self
"""