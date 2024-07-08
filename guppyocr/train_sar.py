'''
THis is the main training code.
'''
import os
from pathlib import Path

import editdistance
import numpy as np
from torch.utils import data

from guppyocr.dataset import dataset
from guppyocr.model.charmap import make_charmap_from_input_dirs
from guppyocr.ocr_dataset import PlainDatasetReader
from guppyocr.train_calamares import TrainingOpts, make_readers

#os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set GPU id at the very begining
import argparse
import random
import math
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.multiprocessing import freeze_support
import pdb
# internal package
from guppyocr.model.sar import sar


from torch import nn




def end_cut(indices, char2id, id2char):
    '''
    indices: numpy array or list of character indices
    charid: char to id conversion
    id2char: id to char conversion
    '''
    cut_indices = []
    for id in indices:
        id = int(id)
        if id != char2id['END']:
            if id != char2id['UNK'] and id != char2id['PAD']:
                cut_indices.append(id2char[id])
        else:
            break
    return ''.join(cut_indices)

def performance_evaluate(pred_choice, target, voc, char2id, id2char, metrics_type):
    '''
    pred_choice: predicted numpy array of [batch_size, seq_len] with index in output_classes
    target: true numpy array of [batch_size, seq_len] with index in output_classes
    voc: vocabular dictionary
    charid: char to id conversion
    id2char: id to char conversion
    metrics_type: evaluation metric name
    '''
    batch_size = target.shape[0]
    predicts = []
    labels = []
    for batch in range(batch_size):
        predict_indices = pred_choice[batch]
        tareget_indices = target[batch]

        predicts.append(end_cut(predict_indices, char2id, id2char))
        labels.append(end_cut(tareget_indices, char2id, id2char))

    if metrics_type == 'accuracy':
        acc_list = [(pred == tar) for pred, tar in zip(predicts, labels)]
        accuracy = 1.0 * sum(acc_list) / len(acc_list)

        return accuracy, acc_list, predicts, labels
    elif metrics_type == 'editdistance':
        ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(predicts, labels)]
        eds = 1.0 * sum(ed_list) / len(ed_list)

        return eds, ed_list, predicts, labels

    return -1

# main function:
if __name__ == '__main__':
    os.nice(10)
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch', type=int, default=2, help='input batch size')
    parser.add_argument(
        '--worker', type=int, default=2, help='number of data loading workers')
    parser.add_argument(
        '--epoch', type=int, default=250, help='number of epochs')
    parser.add_argument('--output', type=str, default='str', help='output folder name')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='svt', help="dataset type - svt|iiit5k|syn90k|synthtext|mysynth")
    parser.add_argument("--val_dataset", type=str, required=True, help="Validation Dataset")
    parser.add_argument('--gpu', type=bool, default=False, help="GPU being used or not")
    parser.add_argument('--metric', type=str, default='accuracy', help="evaluation metric - accuracy|editdistance")

    opt = parser.parse_args()
    print(opt)
    
    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed:", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
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
    
    # set training parameters
    batch_size = opt.batch
    Height = 48
    Width = 640
    feature_height = Height // 4
    feature_width = Width // 8
    Channel = 3
    print("Making charpmap...")
    cmap = make_charmap_from_input_dirs([str(Path(opt.dataset)/"")])
    voc, char2id, id2char = cmap.charmap, cmap.char2id, cmap.id2char
    output_classes = len(voc)
    print(f"Charmap created: {voc}")
    print(f"Length: {len(voc)}")
    embedding_dim = 512
    hidden_units = 512
    layers = 2
    keep_prob = 1.0
    seq_len = 80
    epochs = opt.epoch
    worker = opt.worker
    dataset_path = opt.dataset
    dataset_type = opt.dataset_type
    output_path = opt.output
    trained_model_path = opt.model
    eval_metric = opt.metric

    reader_train = PlainDatasetReader(Path(opt.dataset))
    reader_val = PlainDatasetReader(Path(opt.val_dataset))

    print("Loading Datasets")
    train_dataset = dataset.MySimpleDataset(
        reader_train, Height, Width, seq_len, char2id, augment=True, putend=True)
    test_dataset = dataset.MySimpleDataset(
        reader_val, Height, Width, seq_len, char2id, augment=False, putend=True)
   
    # make dataloader
    train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=int(worker))
        
    test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=int(worker))

    print("Length of train dataset is:", len(train_dataset))
    print("Length of test dataset is:", len(test_dataset))
    print("Number of output classes is:", train_dataset.output_classes)

    # make model output folder
    try:
        os.makedirs(output_path)
    except OSError:
        pass

    # create model
    print("Create model......")
    model = sar(Channel, feature_height, feature_width, embedding_dim, output_classes, hidden_units, layers, keep_prob, seq_len, device)

    if trained_model_path != '':
        if torch.cuda.is_available() == True and opt.gpu == True:
            model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage), strict=False)
            model = torch.nn.DataParallel(model).to(device)
        else:
            model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage), strict=False)
    else:
        if torch.cuda.is_available() == True and opt.gpu == True:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lmbda = lambda epoch: 0.9**(epoch // 300) if epoch < 13200 else 10**(-2)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

    num_batch = math.ceil(len(train_dataset) / batch_size)

    # train, evaluate, and save model
    print("Training starts......")
    if eval_metric == 'accuracy':
        best_acc = float('-inf')
    elif eval_metric == 'editdistance':
        best_acc = float('inf')
    else:
        print("Wrong --metric argument, set it to default")
        eval_metric = 'accuracy'
        best_acc = float('-inf')

    for epoch in range(epochs):
        M_list = []
        for i, data in enumerate(train_dataloader):
            x = data[0] # [batch_size, Channel, Height, Width]
            y = data[1] # [batch_size, seq_len, output_classes]
            x, y = x.to(device), y.to(device)
            #print(x.shape, y.shape)
            optimizer.zero_grad()
            model = model.train()
            predict, _, _, _ = model(x, y)
            #print("Predict size: ", predict.size())
            #print("Target size: ", target.size())
            target = y.max(2)[1] # [batch_size, seq_len]
            #print("Prediction size is:", predict.shape)
            #print("Attention weight size is:", att_weights.shape)
            predict_reshape = predict.permute(0,2,1) # [batch_size, output_classes, seq_len]
            loss = F.nll_loss(predict_reshape, target)
            loss.backward()
            optimizer.step()
            # prediction evaluation
            pred_choice = predict.max(2)[1] # [batch_size, seq_len]
            metric, metric_list, predict_words, labeled_words = performance_evaluate(pred_choice.detach().cpu().numpy(), target.detach().cpu().numpy(), voc, char2id, id2char, eval_metric)
            M_list += metric_list
            print('[Epoch %d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), metric))
            #print("predict prob:", predict[0][0])
            #print("predict words:", predict_words[0])
            #print("labeled words:", labeled_words[0])
        train_acc = float(sum(M_list)/len(M_list))
        print("Epoch {} average train accuracy: {}".format(epoch, train_acc))

        scheduler.step()

        # Validation
        print("Testing......")
        with torch.set_grad_enabled(False):
            M_list = []
            CER_list = []
            for i, data in enumerate(test_dataloader):
                x = data[0] # [batch_size, Channel, Height, Width]
                y = data[1] # [batch_size, seq_len, output_classes]
                x, y = x.to(device), y.to(device)
                model = model.eval()
                predict, _, _, _ = model(x, y)
                # prediction evaluation
                pred_choice = predict.max(2)[1] # [batch_size, seq_len]
                target = y.max(2)[1] # [batch_size, seq_len]
                for bp, bt in zip(pred_choice, target):
                    p_str = end_cut(bp.cpu(), char2id, id2char)
                    tgt_str = end_cut(bt.cpu(), char2id, id2char)
                    cer = editdistance.distance(p_str,tgt_str) / max(1,len(tgt_str))
                    CER_list.append(cer)
                metric, metric_list, predict_words, labeled_words = performance_evaluate(pred_choice.detach().cpu().numpy(), target.detach().cpu().numpy(), voc, char2id, id2char, eval_metric)
                M_list += metric_list
            test_acc = float(sum(M_list)/len(M_list))
            #print("Test predict words:", predict_words[0])
            #print("Test labeled words:", labeled_words[0])
            print("Epoch {} average test accuracy: {}".format(epoch, test_acc))
            print("Epoch {} average test CER: {}".format(epoch, float(np.mean(CER_list))))
            with open(os.path.join(output_path,'statistics.txt'), 'a') as f:
                f.write("{} {}\n".format(train_acc, test_acc))
                f.write(f"Epoch {epoch} Average CER {float(np.mean(CER_list))}\n")

            if eval_metric == 'accuracy':
                if test_acc >= best_acc:
                    print("Save current best model with accuracy:", test_acc)
                    best_acc = test_acc
                    if torch.cuda.is_available() == True and opt.gpu == True:
                        torch.save(model.module.state_dict(), '%s/model_best.pth' % (output_path))
                    else:
                        torch.save(model.state_dict(), '%s/model_best.pth' % (output_path))
            elif eval_metric == 'editdistance':
                if test_acc <= best_acc:
                    print("Save current best model with accuracy:", test_acc)
                    best_acc = test_acc
                    if torch.cuda.is_available() == True and opt.gpu == True:
                        torch.save(model.module.state_dict(), '%s/model_best.pth' % (output_path))
                    else:
                        torch.save(model.state_dict(), '%s/model_best.pth' % (output_path))
    print("Best test accuracy is:", best_acc)