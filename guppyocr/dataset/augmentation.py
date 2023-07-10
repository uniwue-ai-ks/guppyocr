import torch
import torch.utils
import torch.utils.data as data

import albumentations as A

class AugmentedDataset(data.Dataset):
    def __init__(self, base: data.Dataset):
        self.base = base
        

    def __len__(self):
        return len(self.base)
    
    def __getitem__(self,idx):
        image,lbl = self.base[idx]
        image = self.transform(image=image)["image"]
        return image,lbl

    