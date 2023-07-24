'''
THis is the main training code.
'''
import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import editdistance
import numpy as np
import tqdm
from adabelief_pytorch import AdaBelief
from fast_ctc_decode import viterbi_search
from torch.utils import data

#os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set GPU id at the very begining
import argparse
import random
import math
import torch
import torch.nn.parallel
import torch.utils.data
from torch.multiprocessing import freeze_support
# internal package
from guppyocr.dataset import dataset

from torch import nn

from guppyocr.model.charmap import make_charmap_from_input_dirs, make_charmap_from_readers
from guppyocr.model.model import ModelParameters, ModelConfiguration, adjust_voc_model
from guppyocr.ocr_dataset import IDatasetReader, make_dataset_reader


def decode_ctc(pred, id2char) -> str:
    gt_chars = []
    pred_chars = []
    alphabet = [0] * len(id2char)
    net_out = pred.cpu().numpy()
    #print(net_out.shape)
    for k,v in id2char.items():
        alphabet[k] = v
    #print("Alph: ", len(alphabet))
    #print(alphabet)
    seq, _ = viterbi_search(net_out, alphabet)
    seq_out = []
    for x in seq:
        if x == "PAD" or x == "UNK" or x == "END":
            continue
        seq_out.append(x)
    return "".join(seq)

def decode_gt(gt, id2char) -> str:
    gt_chars = []
    for x in gt:
        trans = id2char[int(x)]
        if trans == "PAD" or trans == "UNK" or trans == "END":
            continue
        gt_chars.append(trans)
    return "".join(gt_chars)

def test_model(model, test_dataloader):
    print_indices = set(range(len(test_dataloader)))
    with torch.set_grad_enabled(False):
        CER_SUM = []
        total_chars: int = 0
        errors: int = 0

        for i, data in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            x = data[0] # [B, C, H, W]
            y = data[1] # [batch_size, seq_len, output_classes]
            x, y = x.to(device), y.to(device)
            model = model.eval()
            predict, _, _, _ = model(x, None)

            target = y.max(2)[1] # [batch_size, seq_len]
            try:
                target_firstpad = (target[0] == char2id["PAD"]).nonzero(as_tuple=True)[0][0]
                target = target[:,:target_firstpad]
            except IndexError as e:
                pass

            decoded_gt = decode_gt(target[0], id2char)
            decoded_tgt = decode_ctc(predict[0], id2char)
            edit_distance = editdistance.distance(decoded_tgt,decoded_gt)
            CER = edit_distance / max(1,len(decoded_gt))
            total_chars += max(1,len(decoded_gt))
            errors += edit_distance
            CER_SUM.append(CER)
            if i in print_indices:
                print(f"GT:   {decoded_gt}")
                print(f"Pred: {decoded_tgt}")
                print(f"  CER = {CER:.5f}")

        print("Model pre-test  average CER: {:.5f}".format( float(np.mean(CER_SUM))))
        print("Model pre-test  total CER: {:.5f}".format( errors / total_chars))

class MetricAggregator:
    def __init__(self):
        pass


@dataclass
class TrainingOpts:
    output: str
    dataset: str
    val_dataset: str = ''
    model: str = ''
    test_loaded_model: bool = False

    reader: str = "plain"
    arch: str = "crnn"
    gpu: bool = False
    worker: int = 2
    epoch: int = 100
    grad_clip: bool = False
    early_stopping: int = 1000000
    detailed_results: bool = False


def make_readers(opt: TrainingOpts) -> Tuple[IDatasetReader, IDatasetReader]:
    print("Creating dataset")
    if opt.reader == "plain":
        train_img_path = Path(os.path.join(opt.dataset, 'train'))
        test_img_path = Path(os.path.join(opt.dataset, 'test'))
        if not train_img_path.exists():
            train_img_path = test_img_path = Path(opt.dataset)
        elif not test_img_path.exists():
            test_img_path = train_img_path
    elif opt.reader == "pq":
        train_img_path = Path(opt.dataset)
        if opt.val_dataset:
            test_img_path = Path(opt.val_dataset)
        else:
            test_img_path = train_img_path
    
    #train_img_path = Path("/tmp/tmpout.pq")
    #test_img_path = Path("/tmp/tmpout.pq")
    print(f"Using traing dataset: {train_img_path}")
    print(f"Using val dataset: {test_img_path}")
    reader_train = make_dataset_reader(train_img_path, opt.reader)
    reader_val = make_dataset_reader(test_img_path, opt.reader)
    return reader_train, reader_val



def train_model(opt: TrainingOpts) -> Path:
    manualSeed =  random.randint(1, 10000) # fix seed
    print("Random Seed:", )
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # turn on GPU for models:
    if opt.gpu == False:
        device = torch.device("cpu")
        print("CPU being used!")
    else:
        if torch.cuda.is_available() == True and opt.gpu == True:
            device = torch.device("cuda")
            print("GPU being used!")
        else:
            device = torch.device("cpu")
            print("CPU being used!")


    print(opt)
    print("Creating Datasets")

    reader_train, reader_val = make_readers(opt)

    # build the vocabulary
    cmap = make_charmap_from_readers([reader_train, reader_val])

    voc, char2id, id2char = cmap.charmap, cmap.char2id, cmap.id2char
    print("Training Vocabulary: ", voc)

    # set training parameters
    batch_size = 8
    Height = 48
    Width = 640
    feature_height = Height // 4
    feature_width = Width // 8
    Channel = 3


    output_classes = len(voc)
    hidden_units = 512
    layers = 2
    keep_prob = 1.0
    seq_len = 80
    epochs = opt.epoch
    worker = opt.worker
    dataset_path = opt.dataset
    output_path = opt.output
    trained_model_path = opt.model

    if not Path(output_path).exists():
        print(f"Creating output directory: {output_path}")
        os.makedirs(output_path,exist_ok=True)

    # create model
    print("Create model......")
    mconfig = ModelConfiguration(ModelParameters(Height,Width,Channel,feature_height,feature_width,seq_len,opt.arch,hidden_units,layers,keep_prob,voc,char2id,id2char,False))


    if trained_model_path != '':
        print("Loading Trained Model")
        mconfig, model = ModelConfiguration.load_model(trained_model_path,device)

        did_ajust, result = adjust_voc_model(voc,mconfig,model, device)
        mconfig, model = result.mconfig, result.model

        if did_ajust:
            print("Adjusted model encoding... More training is required")

        voc = mconfig.mconfig.voc
        id2char = mconfig.mconfig.id2char
        char2id = mconfig.mconfig.char2id

        #if torch.cuda.is_available() == True and opt.gpu == True:
        #    model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage), strict=False)
        #    model = torch.nn.DataParallel(model).to(device)
        #else:
        #    model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage), strict=False)
    else:
        model = mconfig.make_model(device)
        if torch.cuda.is_available() == True and opt.gpu == True:
            model = model.to(device)
        else:
            model = model.to(device)

    if torch.cuda.is_available() == True and opt.gpu == True:
        model = torch.nn.DataParallel(model).to(device)


    print("Loading Datasets")
    train_dataset = dataset.MySimpleDataset(reader_train, Height, Width, seq_len, char2id)
    test_dataset = dataset.MySimpleDataset(reader_val, Height, Width, seq_len, char2id)

    # make dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(worker))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(worker))

    print("Length of train dataset is: ", len(train_dataset))
    print("Length of test dataset is: ", len(test_dataset))
    print("Number of output classes is: ", train_dataset.output_classes)

    # make model output folder
    try:
        os.makedirs(output_path)
    except OSError:
        pass



    if opt.test_loaded_model and trained_model_path:
        print("Testing trained model")
        test_model(model, test_dataloader)

    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = AdaBelief(model.parameters(), print_change_log=False)
    #optimizer = SGD(model.parameters(),lr=0.001)
    lmbda = lambda epoch: 0.9**(epoch // 300) if epoch < 100 else 10**(-2)
    if False:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    else:
        scheduler = None

    num_batch = math.ceil(len(train_dataset) / batch_size)

    print("Beginning training")
    best_total_cer = float("inf")
#   model.module.rnn[-1].
    epochs_without_improvement = 0
    detailed_results = []
    for epoch in range(epochs):
        loss_avg = []
        progress = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, data in progress:
            x = data[0] # [batch_size, Channel, Height, Width]

            y = data[1] # [batch_size, seq_len, output_classes]
            x, y = x.to(device), y.to(device)
            #show_images([x[0].cpu().permute(1,2,0).numpy()], ["Hello"])
            #print(x.shape, y.shape)
            optimizer.zero_grad()
            model = model.train()
            predict, _, _, _ = model(x, y)
            #print("Predict size: ", predict.size())
            #print("Target size: ", y.size())
            target = y.max(2)[1] # [batch_size, seq_len]
            #print(target.cpu().numpy())
            try:
                target_firstpad = (target == char2id["PAD"]).cpu().numpy().argmax(axis=1).tolist()
                target_firstpad = [(x if x > 0 else seq_len) for x in target_firstpad ]
                #print(target_firstpad)
                #sys.exit(1)
                #firstpads = [ int(x[0][0]) for x in target_firstpad.cpu().numpy() ]
                #print(firstpads)
                #target = target[:,:target_firstpad]
            except IndexError as e:
                pass
           # print(target.cpu().numpy())
            #print(target_firstpad)

            #show_images([np.transpose(x[0].cpu().numpy(),(1,2,0))],[decode_gt(target[0].cpu().numpy(),id2char)])
            #print("Prediction size is:", predict.shape)
            predict_reshape = predict.permute(1,0,2) # [seq_len, batch_size, output_classes]
            #print("Predict Reshaped: ", predict_reshape.size())
            #print("Target", target.size())
            predict_softmax = predict_reshape.log_softmax(2)
            input_lengths = tuple([int(predict_reshape.size()[0])] * target.size()[0])
            #target_lengths = #tuple([target.size()[1]] * batch_size)
            target_lengths = target_firstpad
            #print("Inp Len: ", input_lengths)
            #print("Target Len: ", target_lengths)
            loss = nn.CTCLoss(blank=char2id["PAD"],reduction="sum").forward(log_probs=predict_softmax,
                                                            targets=target,
                                                            input_lengths=input_lengths,
                                                            target_lengths=target_lengths)
            loss_avg.append(float(loss))


            loss.backward()
            if opt.grad_clip:
                """
                total_norm = 0
                for p in model.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Gradient norm: {total_norm:.3f}")
                """
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # prediction evaluation
            pred_choice = predict.max(2)[1] # [batch_size, seq_len]

            progress.desc = f'[Epoch {epoch}: {i}/{num_batch}] train loss: {np.mean(loss_avg)}'
            #print("predict prob:", predict[0][0])
            #print("predict words:", predict_words[0])
            #print("labeled words:", labeled_words[0])
        print("Epoch {} average loss: {:.5f} ".format(epoch, float(sum(loss_avg) / len(loss_avg))))

        if scheduler is not None:
            scheduler.step()
        
        def print_val(gt, pred, id2char):
            # reverse char2id
            #id2char = {v:k for k,v in id2char.items()}
            gt_chars = []
            pred_chars = []
            alphabet = [0] * len(id2char)
            net_out = pred.cpu().numpy()
            print(net_out.shape)
            for k,v in id2char.items():
                alphabet[k] = v
            print("Alph: ", len(alphabet))
            print(alphabet)
            seq, _ = viterbi_search(net_out, alphabet)
            for x in gt:
                gt_chars.append(id2char[int(x)])
            print("GT: " + "".join(gt_chars))
            print("Pred ", seq)


        if epoch < 20:
            #continue
            pass
        # Validation
        print("Testing......")
        print_indices = set(random.choices(list(range(len(test_dataloader))),k=10))
        with torch.set_grad_enabled(False):
            CER_SUM = []
            total_chars: int = 0
            errors: int = 0

            for i, data in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                x = data[0] # [batch_size, Channel, Height, Width]
                y = data[1] # [batch_size, seq_len, output_classes]
                x, y = x.to(device), y.to(device)
                model = model.eval()
                predict, _, _, _ = model(x, y)
                # prediction evaluation
                pred_choice = predict.max(2)[1] # [batch_size, seq_len]
                pred_decoded = predict.argmax(dim=2)
                target = y.max(2)[1] # [batch_size, seq_len]
                try:
                    target_firstpad = (target[0] == char2id["PAD"]).nonzero(as_tuple=True)[0][0]
                    target = target[:,:target_firstpad]
                except IndexError as e:
                    pass
                #print(target_firstpad)
                #if i % 64 == 0:
                #    print_val(target[0], predict[0], id2char)
                decoded_gt = decode_gt(target[0], id2char)
                decoded_tgt = decode_ctc(predict[0], id2char)
                edit_distance = editdistance.distance(decoded_tgt,decoded_gt)
                CER = edit_distance / max(1,len(decoded_gt))
                total_chars += max(1,len(decoded_gt))
                errors += edit_distance
                CER_SUM.append(CER)
                if i in print_indices:
                    print(f"GT:   {decoded_gt}")
                    print(f"Pred: {decoded_tgt}")
                    print(f"  CER = {CER:.5f}")

            print("Epoch {} average CER: {:.5f}".format(epoch, float(np.mean(CER_SUM))))
            print("Epoch {} total CER: {:.5f}".format(epoch, errors / total_chars))
            with open(os.path.join(output_path,'statistics.txt'), 'a') as f:
                f.write("TEST CER Epoch {} {}\n".format(epoch, float(np.mean(CER_SUM))))
                f.write("Epoch {} total CER: {:.5f}\n".format(epoch, errors / total_chars))
                f.write("Best CER (until now): {:.5f}\n".format(best_total_cer))

            total_cer = errors / total_chars
            if total_cer < best_total_cer:
                best_total_cer = total_cer
                state_dict = None
                if torch.cuda.is_available() == True and opt.gpu == True:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                model_output_path = Path(output_path) / "model_best.pth"
                mconfig.save(model_output_path, state_dict)
                print(f"Saving model as {model_output_path}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if opt.detailed_results:
                state_dict = None
                if torch.cuda.is_available() == True and opt.gpu == True:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                mconfig.save(Path(output_path) / f"m_epoch{epoch:04}.pth", state_dict)
                detailed_results.append({
                    "Epoch": int(epoch),
                    "TrainLoss": float(np.mean(loss_avg)),
                    "ValTotalCER": float(total_cer)
                })
            if epochs_without_improvement > opt.early_stopping:
                break

    print("Best total CER is:", best_total_cer)
    if detailed_results:
        print("Saving detailed results as CSV")
        import pandas as pd
        df = pd.DataFrame(detailed_results)
        df.to_csv(Path(output_path) / "detailed_results.csv")


    return Path(output_path) / "model_best.pth"

# main function:
if __name__ == '__main__':
    os.nice(10)
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--worker', type=int, default=2, help='number of data loading workers')
    parser.add_argument(
        '--epoch', type=int, default=250, help='number of epochs')
    parser.add_argument('--output', type=str, default='str', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path for finetuning')
    parser.add_argument('--reader', type=str, default='plain', help="Reader Module to use for reading the dataset")
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--val_dataset', type=str, required=False, help="validation dataset path")
    parser.add_argument('--gpu', type=bool, default=False, help="Use GPU")
    parser.add_argument("--grad_clip", action="store_true", help="Enable gradient clipping (default off)")
    parser.add_argument("--arch", type=str, choices=["crnn","calamari", "stncrnn", "trcrnn"])
    parser.add_argument("--test_loaded_model", action="store_true")
    parser.add_argument("--detailed", action="store_true", help="Enable detailed output mode")


    opt = parser.parse_args()

    training_opts = TrainingOpts(
        output=opt.output,
        dataset=opt.dataset,
        val_dataset=opt.val_dataset,
        model=opt.model,
        test_loaded_model=opt.test_loaded_model,
        reader=opt.reader,
        arch=opt.arch,
        gpu=opt.gpu,
        worker=opt.worker,
        epoch=opt.epoch,
        grad_clip=opt.grad_clip,
        detailed_results=opt.detailed
    )
    train_model(training_opts)



    