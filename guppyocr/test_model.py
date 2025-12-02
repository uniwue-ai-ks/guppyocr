from csv import Dialect
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set GPU id at the very begining
import argparse
from pathlib import Path
from typing import Optional
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
from guppyocr.ocr_dataset import IDatasetReader, PlainDatasetReader, SplitDatasetAdaptor, make_dataset_reader

from guppyocr.train_calamares import test_model



def test_model_raw(model_path: Path, dataset_path: Path, reader: str, csv_path: Optional[Path] = None, json_path: Optional[Path] = None):
    reader: PlainDatasetReader = make_dataset_reader(Path(dataset_path))

    # load the model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU being used!")
    else:
        device = torch.device("cpu")
        print("CPU being used!")

    mconfig, model = ModelConfiguration.load_model(model_path, device)

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
        num_workers=4)
    
    with torch.set_grad_enabled(False):
        test_result = test_model(mconfig,model,test_dataloader,device,8,0,0,None,None,True, store_results=True if args.csv or args.json else False)

    print(f"Total test CER: {test_result.total_cer}")
    if csv_path:
        import csv
        # open the csv
        lines = []
        for fname, gt in zip(reader.list_files(),test_result.results):
            with open(csv_path, "w", newline='') as of:
                writer = csv.writer(of,delimiter=";", doublequote=True)
                writer.writerow((fname,gt))

    if json_path:
        import json
        lines = [ {"file": fname, "gt": gt, "pred": pred} for (fname,gt, pred) in zip(reader.list_files(),[reader.get_text(i) for i in range(len(reader))], test_result.results)]

        with open(json_path,"w") as of:
            json.dump(lines,of)

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
    #parser.add_argument("--worker", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--csv", type=str, required=False)
    parser.add_argument("--json", type=str, required=False)

    
    args = parser.parse_args()
    test_model_raw(model_path=Path(args.model), dataset_path=Path(args.dataset), reader=args.reader,csv_path=Path(args.csv) if args.csv else None,json_path=Path(args.json) if args.json else None)
   

