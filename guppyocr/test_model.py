import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set GPU id at the very begining
import argparse
from pathlib import Path
import torch
import torch.nn.parallel
import torch.utils.data
from torch.multiprocessing import freeze_support
from torch.utils.tensorboard import SummaryWriter
# internal package
from guppyocr.dataset import dataset

from torch import is_grad_enabled, nn

from guppyocr.model.charmap import make_charmap_from_input_dirs, make_charmap_from_readers
from guppyocr.model.model import ModelParameters, ModelConfiguration, adjust_voc_model
from guppyocr.ocr_dataset import IDatasetReader, SplitDatasetAdaptor, make_dataset_reader

from guppyocr.train_calamares import test_model

if __name__ == '__main__':
    os.nice(10)
    freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='',
                        help='model path')
    parser.add_argument('--reader', type=str, default='plain',
                        help="Reader Module to use for reading the dataset")
    parser.add_argument('--dataset', type=str,
                        required=True, help="dataset path")
    parser.add_argument("--worker", type=int, default=4, help="Number of data loader workers")
    
    args = parser.parse_args()
    reader = make_dataset_reader(Path(args.dataset))

    # load the model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU being used!")
    else:
        device = torch.device("cpu")
        print("CPU being used!")

    mconfig, model = ModelConfiguration.load_model(args.model, device)

    # check if the VOC of the 
    cmap, _ = make_charmap_from_readers([reader],False)
    if not set(cmap.charmap).issubset(set(mconfig.mconfig.char2id.keys())):
        print("Warning: Test set contains chars not in model set")
        print(f"Missing Chars: {''.join(set(cmap.charmap) - set(mconfig.mconfig.char2id.keys()))}")



    test_dataset = dataset.MySimpleDataset(
        reader, mconfig.mconfig.Height, mconfig.mconfig.Width, mconfig.mconfig.seq_len, mconfig.mconfig.char2id, augment=False)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=int(args.worker))
    
    with torch.set_grad_enabled(False):
        test_result = test_model(mconfig,model,test_dataloader,device,8,0,0,None,None,True)

    print(f"Total test CER: {test_result.total_cer}")


