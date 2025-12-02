'''
THis is the main training code.
'''
import itertools
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Collection, List, Dict, NamedTuple, Tuple, Optional

import editdistance
import numpy as np
import tqdm
from adabelief_pytorch import AdaBelief
from fast_ctc_decode import viterbi_search
from torch.utils import data

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set GPU id at the very begining
import argparse
import random
import math
import torch
import torch.nn.parallel
import torch.utils.data
from torch.multiprocessing import freeze_support
from torch.utils.tensorboard import SummaryWriter
# internal package
from guppyocr.dataset import dataset

from torch import is_grad_enabled, nn

from guppyocr.model.charmap import make_charmap_from_input_dirs, make_charmap_from_readers, merge_charmaps
from guppyocr.model.model import ModelParameters, ModelConfiguration, adjust_voc_model
from guppyocr.ocr_dataset import IDatasetReader, SplitDatasetAdaptor, make_dataset_reader


def decode_ctc(pred, id2char) -> str:
    gt_chars = []
    pred_chars = []
    alphabet = [0] * len(id2char)
    net_out = pred.cpu().numpy()
    # print(net_out.shape)
    for k, v in id2char.items():
        alphabet[k] = v
    # print("Alph: ", len(alphabet))
    # print(alphabet)
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

class MetricAggregator:
    def __init__(self):
        pass


@dataclass
class TrainingOpts:
    output: str
    dataset: str
    val_dataset: str = ''
    split_val: Optional[float] = None
    model: str = ''
    test_loaded_model: bool = False

    reader: str = "plain"
    arch: str = "crnn"
    gpu: bool = False
    worker: int = 2
    epoch: int = 100
    grad_clip: bool = False
    early_stopping: int = 1000000

    optimizer: str = "AdaBelief"  # Alternative SGD, Adam, AdaBelief
    lr: float = 0.001
    grad_clip: bool = True
    batch_size: int = 8
    test_ignore_chars: List[str] = field(default_factory=lambda: [])
    augment: bool = False
    detailed_results: bool = False

    img_width: int = 640
    img_height: int = 48
    channel: int = 3

    hidden_units: int = 512
    layers: int = 2
    keep_prob: float = 1.0
    seq_len: int = 80
    
    save_every_epoch: bool = False
    crossfold: int = 0
    fold: int = 0

    l2_regularizer: Optional[float] = None
    #add_chars: List[str] = field(default_factory=lambda: list("1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ[]().,?!\""))
    add_chars: List[str] = field(default_factory=list)

    def get_hyperparameters(self) -> dict:
        hparams_dict = asdict(self)
        hparams_dict["add_chars"] = "".join(sorted(hparams_dict["add_chars"]))
        hparams_dict["test_ignore_chars"] = "".join(sorted(hparams_dict["test_ignore_chars"]))
        return hparams_dict


def make_readers(opt: TrainingOpts) -> Tuple[IDatasetReader, IDatasetReader]:
    print("Creating dataset")
    use_val_dataset = opt.val_dataset

    if opt.reader == "memory":
        return opt.dataset, opt.val_dataset

    if opt.reader == "plain":
        if opt.val_dataset:
            train_img_path = Path(opt.dataset)
            val_img_path = Path(opt.val_dataset)
            use_val_dataset = True
        else:
            train_img_path = Path(os.path.join(opt.dataset, 'train'))
            val_img_path = Path(os.path.join(opt.dataset, 'test'))
            if not train_img_path.exists():
                train_img_path = val_img_path = Path(opt.dataset)
                use_val_dataset = False
            elif not val_img_path.exists():
                val_img_path = train_img_path
                use_val_dataset = False
    elif opt.reader == "pq":
        train_img_path = Path(opt.dataset)
        if opt.val_dataset:
            val_img_path = Path(opt.val_dataset)
        else:
            val_img_path = train_img_path

    # train_img_path = Path("/tmp/tmpout.pq")
    # test_img_path = Path("/tmp/tmpout.pq")
    print(f"Using traing dataset: {train_img_path}")
    print(f"Using val dataset: {val_img_path}")
    reader_train = make_dataset_reader(train_img_path, opt.reader)
    if use_val_dataset and opt.split_val is not None:
        raise RuntimeError("Cannot specify --split_val and a validation dataset at the same time")
    elif opt.split_val is not None:
        reader_train, reader_val = SplitDatasetAdaptor.split_reader(reader_train,opt.split_val,seed=123456)
    else:
        reader_val = make_dataset_reader(val_img_path, opt.reader)

    return reader_train, reader_val

@dataclass
class TrainingResult:
    best_model_path: Path
    best_val_cer: float


def make_crossfold(train_ds: data.Dataset, val_ds: data.Dataset, opts: TrainingOpts) -> Tuple[data.Dataset, data.Dataset]:
    if opts.crossfold <= 0: return train_ds, val_ds
    # else, check if this is viable
    if opts.val_dataset: raise RuntimeError("Cannot specify val_dataset with crossfold training")
    if opts.split_val is not None: raise RuntimeError("Cannot specify split_val with crossfold training")

    # okay, make a subset for training and validation
    all_indices = list(range(len(train_ds)))
    rng = random.Random(123)
    rng.shuffle(all_indices)

    len_part = len(all_indices) // opts.crossfold

    parts = []
    for fold_i in range(opts.crossfold):
        if fold_i != opts.crossfold - 1:
            parts.append(all_indices[fold_i * len_part: (fold_i+1) * len_part])
        else:
            parts.append(all_indices[fold_i * len_part : ])
    
    new_val_ds = data.Subset(train_ds,parts.pop(opts.fold))
    new_train_ds = data.Subset(train_ds, list(itertools.chain.from_iterable(parts)))
    
    print(f"Using crossfold training ({opts.fold} / {opts.crossfold}) with {len(new_train_ds)} train and {len(new_val_ds)} validation samples")

    return new_train_ds, new_val_ds


class ResultTuple(NamedTuple):
    pred: str
    gt: str
    cer: float
    correct: int

class TestModelResult(NamedTuple):
    mean_cer: float
    total_cer: float
    accuracy: float
    results: Optional[List[ResultTuple]] = None

def test_model(mc: ModelConfiguration, model: torch.nn.Module, test_dataloader: torch.utils.data.DataLoader, 
               device, batch_size, epoch: int, num_batch: int,
               test_ignore_chars: Optional[Collection[str]]= None,
               writer: Optional[SummaryWriter] = None, print_random_indices: bool = True, store_results: bool = False) -> TestModelResult:
    
    assert not torch.is_grad_enabled(), "For testing the model, autograd should be turned off"
    char2id, id2char = mc.mconfig.char2id, mc.mconfig.id2char

    if test_ignore_chars is None:
        test_ignore_chars = set()
    
    print_indices = set(random.choices(
            list(range(len(test_dataloader)*batch_size)), k=10))
    
    CER_SUM = []
    total_chars: int = 0
    errors: int = 0
    model = model.eval()
    result_strings = []

    for i, data in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        x = data[0]  # [batch_size, Channel, Height, Width]
        y = data[1]  # [batch_size, seq_len, output_classes]
        #y_np = y.numpy()
        x, y_gpu = x.to(device), y.clone().to(device)

        predict, _, _, _ = model(x, y_gpu)
        # prediction evaluation
        #pred_choice = predict.max(2)[1]  # [batch_size, seq_len]
        #pred_decoded = predict.argmax(dim=2)
        target = (y.max(2)[1])  # [batch_size, seq_len]
        for batch_i in range(int(x.shape[0])):
            this_target = target[batch_i]
            try:
                target_firstpad = (this_target == char2id["PAD"]).nonzero(as_tuple=True)[0][0]
                this_target = this_target[:target_firstpad]
            except IndexError as e:
                pass

            decoded_gt = decode_gt(this_target, id2char)
            decoded_tgt = decode_ctc(predict[batch_i], id2char)
            decoded_gt_eval = decoded_gt
            decoded_tgt_eval = decoded_tgt
            
            
            for c in test_ignore_chars:
                decoded_gt_eval = decoded_gt_eval.replace(c, '')
                decoded_tgt_eval = decoded_tgt_eval.replace(c, '')
            # TODO: should spaces at the beginning / end should also by default be ignored (ie. stripping the output here?)
            edit_distance = editdistance.distance(
                decoded_tgt_eval, decoded_gt_eval)
            
            

            CER = edit_distance / max(1, len(decoded_gt_eval))
            if store_results:
                result_strings.append(ResultTuple(decoded_tgt,decoded_gt,CER,CER == 0.0))
            total_chars += max(1, len(decoded_gt_eval))
            errors += edit_distance
            CER_SUM.append(CER)
            if batch_i + i * batch_size in print_indices:
                if print_random_indices:
                    print(f"GT:   {decoded_gt}")
                    print(f"Pred: {decoded_tgt}")
                    print(f"  CER = {CER:.5f}")
                # write example to tensorboard
                if writer:
                    writer.add_text(
                        f"GT_{i}", decoded_gt, epoch * num_batch)
                    writer.add_text(
                        f"Pred_{i}", decoded_tgt, epoch * num_batch)
    CER_arr = np.array(CER_SUM) 
    return TestModelResult(float(np.mean(CER_arr)),float(errors / total_chars),float(np.count_nonzero(CER_arr == 0.0) / CER_arr.shape[0]),result_strings if store_results else None)




def train_model(opt: TrainingOpts, seed: int = None) -> TrainingResult:
    writer = SummaryWriter()

    if seed is not None:
        manualSeed = seed
    else:
        manualSeed = random.randint(1, 10000)  # fix seed
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
    
    cmap_train, cmap_sl_train = make_charmap_from_readers([reader_train], use_tqdm=True,add_chars=opt.add_chars)
    cmap_test, cmap_sl_test  = make_charmap_from_readers([reader_val], use_tqdm=True)

    if not set(cmap_train.charmap).issuperset(set(cmap_test.charmap)):
        print("WARNING: Test set contains Chars not available in train")
        print(f"Missing Chars: {''.join(set(cmap_train.charmap) - set(cmap_test.charmap))}")
    
    cmap = merge_charmaps([cmap_train, cmap_test])
    max_seq_len_ds = max(cmap_sl_train, cmap_sl_test)
    del cmap_train, cmap_test, cmap_sl_test, cmap_sl_train

    #cmap, max_seq_len_ds = make_charmap_from_readers(
    #    [reader_train, reader_val],use_tqdm=True)

    voc, char2id, id2char = cmap.charmap, cmap.char2id, cmap.id2char
    print("Training Vocabulary: ", voc)

    # set training parameters
    batch_size = opt.batch_size
    Height = opt.img_height
    Width = opt.img_width
    # Width = 640
    # Width = 1280
    feature_height = Height // 4
    feature_width = Width // 8
    Channel = opt.channel

    output_classes = len(voc)
    hidden_units = opt.hidden_units
    layers = opt.layers
    keep_prob = opt.keep_prob
    seq_len = opt.seq_len
    epochs = opt.epoch
    worker = opt.worker
    dataset_path = opt.dataset
    output_path = opt.output
    trained_model_path = opt.model

    if max_seq_len_ds > seq_len:
        seq_len = max_seq_len_ds + 2
        print(f"Warning: Changing Max Sequence Length to: {seq_len}")

    if not Path(output_path).exists():
        print(f"Creating output directory: {output_path}")
        os.makedirs(output_path, exist_ok=True)

    # create model
    print("Create model......")
    mconfig = ModelConfiguration(ModelParameters(Height, Width, Channel, feature_height, feature_width,
                                 seq_len, opt.arch, hidden_units, layers, keep_prob, voc, char2id, id2char, False))

    if trained_model_path != '':
        print("Loading Trained Model Weights")
        mconfig, model = ModelConfiguration.load_model(
            trained_model_path, device)

        did_ajust, result = adjust_voc_model(voc, mconfig, model, device)
        mconfig, model = result.mconfig, result.model

        if did_ajust:
            print("Adjusted model encoding. More training is required")

        voc = mconfig.mconfig.voc
        id2char = mconfig.mconfig.id2char
        char2id = mconfig.mconfig.char2id

        # if torch.cuda.is_available() == True and opt.gpu == True:
        #    model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage), strict=False)
        #    model = torch.nn.DataParallel(model).to(device)
        # else:
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
    train_dataset = dataset.MySimpleDataset(
        reader_train, Height, Width, seq_len, char2id, augment=opt.augment)
    test_dataset = dataset.MySimpleDataset(
        reader_val, Height, Width, seq_len, char2id, augment=False)

    if opt.crossfold > 0:
        train_dataset, test_dataset = make_crossfold(train_dataset,test_dataset,opt)
        train_dataset.output_classes = train_dataset.dataset.output_classes
    
    # make dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(worker))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
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
        with torch.set_grad_enabled(False):
            test_loaded_model_res = test_model(mconfig, model, test_dataloader, device, batch_size,0,0,opt.test_ignore_chars,None,True)
            print(f"Loaded Model test CER: {test_loaded_model_res.total_cer :.4f}")

    if opt.optimizer.lower() == "adabelief":
        optimizer = AdaBelief(model.parameters(),
                              print_change_log=False, lr=opt.lr, weight_decay=opt.l2_regularizer if opt.l2_regularizer else 0)
    elif opt.optimizer.lower() == "adam":
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=opt.lr,weight_decay=opt.l2_regularizer)
        assert not opt.l2_regularizer
    elif opt.optimizer.lower() == "sgd":
        import torch.optim as optim
        optimizer = optim.SGD(model.parameters(), lr=opt.lr,weight_decay=opt.l2_regularizer if opt.l2_regularizer else 0)
        #assert not opt.l2_regularizer
    else:
        raise RuntimeError("Invalid optimizer specified")

    def lmbda(epoch): return 0.9**(epoch // 300) if epoch < 100 else 10**(-2)
    if False:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    else:
        scheduler = None
    import torch.optim as optim
    #scheduler = optim.lr_scheduler.CyclicLR(optimizer,0.0001, 0.001,step_size_up=len(train_dataloader) // 2, step_size_down=len(train_dataloader) // 2,cycle_momentum=False)

    num_batch = math.ceil(len(train_dataset) / batch_size)

    # write to tensorboard
    # writer.add_graph(model, torch.rand(1, Channel, Height, Width).to(device))

    print("Beginning training")
    best_total_cer = float("inf")
#   model.module.rnn[-1].
    epochs_without_improvement = 0

    writer.add_hparams(opt.get_hyperparameters(),dict())
    for epoch in range(epochs):
        loss_avg = []
        progress = tqdm.tqdm(enumerate(train_dataloader),
                             total=len(train_dataloader))
        for i, data in progress:
            x = data[0]  # [batch_size, Channel, Height, Width]

            y = data[1]  # [batch_size, seq_len, output_classes]
            x, y = x.to(device), y.to(device)
            # show_images([x[0].cpu().permute(1,2,0).numpy()], ["Hello"])
            # print(x.shape, y.shape)
            optimizer.zero_grad()
            model = model.train()
            predict, _, _, _ = model(x, y)
            # print("Predict size: ", predict.size())
            # print("Target size: ", y.size())
            target = y.max(2)[1]  # [batch_size, seq_len]
            # print(target.cpu().numpy())
            try:
                target_firstpad = (target == char2id["PAD"]).cpu(
                ).numpy().argmax(axis=1).tolist()
                target_firstpad = [(x if x > 0 else seq_len)
                                   for x in target_firstpad]
                # print(target_firstpad)
                # sys.exit(1)
                # firstpads = [ int(x[0][0]) for x in target_firstpad.cpu().numpy() ]
                # print(firstpads)
                # target = target[:,:target_firstpad]
            except IndexError as e:
                pass
           # print(target.cpu().numpy())
            # print(target_firstpad)

            # show_images([np.transpose(x[0].cpu().numpy(),(1,2,0))],[decode_gt(target[0].cpu().numpy(),id2char)])
            # print("Prediction size is:", predict.shape)
            # [seq_len, batch_size, output_classes]
            predict_reshape = predict.permute(1, 0, 2)
            # print("Predict Reshaped: ", predict_reshape.size())
            # print("Target", target.size())
            predict_softmax = predict_reshape.log_softmax(2)
            input_lengths = tuple(
                [int(predict_reshape.size()[0])] * target.size()[0])
            # target_lengths = #tuple([target.size()[1]] * batch_size)
            target_lengths = target_firstpad
            # print("Inp Len: ", input_lengths)
            # print("Target Len: ", target_lengths)

            # TODO: is mean or sum better here? it previously was "sum", but that tends to huge losses
            loss = nn.CTCLoss(blank=char2id["PAD"], reduction="sum", zero_infinity=True).forward(log_probs=predict_softmax,
                                                                                                 targets=target,
                                                                                                 input_lengths=input_lengths,
                                                                                                 target_lengths=target_lengths)
            loss_avg.append(float(loss.detach()))

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

            if scheduler is not None:
                scheduler.step()

            writer.add_scalar('Loss/train', loss.item(), epoch * num_batch + i)

            progress.desc = f"[Epoch {epoch}: {i}/{num_batch}] train loss: {np.mean(loss_avg):.3f}  LR: {optimizer.param_groups[0]['lr']}"
        
        print(f"Epoch {epoch} average loss: {float(np.mean(loss_avg)):.5f} ")

        #if scheduler is not None: # WHY is this here, this should update after every optimizer step (unless per_epoch scheduling ist used)
        #    scheduler.step()

        # Validation
        print("Testing......")
        
        with torch.set_grad_enabled(False):
            val_result = test_model(mconfig,model,test_dataloader,device,batch_size,epoch,num_batch,opt.test_ignore_chars,writer,print_random_indices=True)

            writer.add_scalar('CER/val mean', val_result.mean_cer,
                                epoch * num_batch + i)
            writer.add_scalar('CER/val total', val_result.total_cer,
                                epoch * num_batch + i)
            writer.add_scalar("Line Accuracy", val_result.accuracy)

            #print(f"Epoch {epoch} average CER: {val_result.mean_cer :.5f}") # This is absolutely misleading and not required
            print(f"Epoch {epoch} total CER: {val_result.total_cer :.5f} / Line Accuracy: {val_result.accuracy :.5f}")

            with open(os.path.join(output_path, 'statistics.csv'), 'a') as f:
                f.write(f"{epoch}, {val_result.mean_cer}, {val_result.total_cer}, {best_total_cer}, {val_result.accuracy}\n")
                #f.write("TEST CER Epoch {} {}\n".format(
                #    epoch, )))
                #f.write("Epoch {} total CER: {:.5f}\n".format(
                #    epoch, errors / total_chars))
                #f.write("Best CER (until now): {:.5f}\n".format(
                #    best_total_cer))
            
            def save_model(model_file: Path):
                state_dict = None
                if torch.cuda.is_available() == True and opt.gpu == True:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                mconfig.save(model_file, state_dict)
                print(f"Saving model as {model_file}")

                # save also the optimizer state
                opt_state_path = model_file.with_suffix(".pth.optim")
                torch.save({
                    "optimizer_state_dict": optimizer.state_dict
                }, opt_state_path)
            
            if val_result.total_cer < best_total_cer:
                best_total_cer = val_result.total_cer
                save_model(Path(output_path) / "model_best.pth")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if opt.save_every_epoch:
                os.makedirs(Path(output_path) / "mod_epochs", exist_ok=True)
                save_model(Path(output_path) / "mod_epochs" / f"e_{epoch}.pth")

            if epochs_without_improvement >= opt.early_stopping:
                break

    print("Best total CER is:", best_total_cer)
    return TrainingResult(
        best_model_path=Path(output_path) / "model_best.pth",
        best_val_cer=best_total_cer
    )


# main function:
if __name__ == '__main__':
    os.nice(10)
    freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--worker', type=int, default=2, help='number of data loading workers')
    parser.add_argument(
        '--epoch', type=int, default=250, help='number of epochs')
    parser.add_argument('--output', type=str,
                        default='str', help='output folder')
    parser.add_argument('--model', type=str, default='',
                        help='model path for finetuning')
    parser.add_argument('--reader', type=str, default='plain',
                        help="Reader Module to use for reading the dataset")
    parser.add_argument('--dataset', type=str,
                        required=True, help="dataset path")
    parser.add_argument('--val_dataset', type=str,
                        required=False, help="validation dataset path")
    parser.add_argument('--split_val', type=float, required=False, default=None, help="Percentage of dataset to use for training")
    parser.add_argument('--gpu', type=bool, default=False, help="Use GPU")
    parser.add_argument("--grad_clip", action="store_true",
                        help="Enable gradient clipping (default off)")
    parser.add_argument("--arch", type=str,
                        choices=ModelConfiguration.valid_model_names())
    parser.add_argument("--test_loaded_model", action="store_true")
    parser.add_argument("--detailed", action="store_true",
                        help="Enable detailed output mode")
    parser.add_argument("--augment", action="store_true",
                        default=False, help="Enable image augmentation")
    parser.add_argument('--early_stopping', type=int, default=1000000,
                        help='Early stopping patience (default: 1000000)')

    parser.add_argument('--optimizer', type=str, default='AdaBelief',
                        help='Optimizer type (default: AdaBelief)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    
    parser.add_argument('--detailed_results', action='store_true',
                        help='Enable detailed result logging (optional)')

    parser.add_argument('--img_width', type=int, default=640,
                        help='Image width (default: 640)')
    parser.add_argument('--img_height', type=int, default=48,
                        help='Image height (default: 48)')
    parser.add_argument('--channel', type=int, default=3,
                        help='Number of image channels (default: 3)')

    parser.add_argument('--hidden_units', type=int, default=512,
                        help='Number of hidden units (default: 512)')
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of layers (default: 2)')
    parser.add_argument('--keep_prob', type=float, default=1.0,
                        help='Dropout keep probability (default: 1.0)')
    parser.add_argument('--seq_len', type=int, default=80,
                        help='Sequence length (default: 80)')
    parser.add_argument("--crossfold", type=int, default=0, help="Number of folds")
    parser.add_argument("--fold", type=int, default=0, help="Current fold")
    parser.add_argument("--l2_regularizer", type=float, default=0, help="Set L2 regularization (weight decay)")
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=12345678, help="Seed for the dataset split")

    import sys
    print(sys.argv)
    args = parser.parse_args()
    training_opts = TrainingOpts(
                output=args.output,
        dataset=args.dataset,
        val_dataset=args.val_dataset,
        split_val=args.split_val,
        model=args.model,
        test_loaded_model=args.test_loaded_model,
        reader=args.reader,
        arch=args.arch,
        gpu=args.gpu,
        worker=args.worker,
        epoch=args.epoch,
        grad_clip=args.grad_clip,
        early_stopping=args.early_stopping,
        optimizer=args.optimizer,
        lr=args.lr,
        batch_size=args.batch_size,
        augment=args.augment,
        detailed_results=args.detailed_results,
        img_width=args.img_width,
        img_height=args.img_height,
        channel=args.channel,
        hidden_units=args.hidden_units,
        layers=args.layers,
        keep_prob=args.keep_prob,
        seq_len=args.seq_len,
        crossfold=args.crossfold,
        fold=args.fold,
        l2_regularizer=args.l2_regularizer
    )
    train_model(training_opts, seed=args.seed)

    if args.test_path is not None:
        print("Testing model using test_model.py script")
        from guppyocr.test_model import test_model_raw
        test_model_raw(model_path=Path(args.output) / "model_best.pth",
                       dataset_path=Path(args.test_path),
                       reader="plain",
                       csv_path=Path(args.output) / "test_results.csv",
                       json_path=Path(args.output) / "test_results.json")        
