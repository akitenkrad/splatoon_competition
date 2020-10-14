import os
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .datasets import SplatoonDataset, sdata_to
from .model import SimpleTransformer

def run_test(ds_path: str, weights: str, outfile: str):
    
    epochs = 1
    batch_size = 128
    
    # device
    device = torch.device('cpu')
    
    # load dataset
    dataset = SplatoonDataset(str(Path(ds_path).absolute()))
    
    # read meta counts
    n_lobby_modes = len(dataset.lobby_mode_vocab)
    n_modes = len(dataset.mode_vocab)
    n_weapons = len(dataset.weapon_vocab)
    n_ranks = len(dataset.rank_vocab)
    n_stages = len(dataset.stage_vocab)
    
    # split dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # prepare model
    model = SimpleTransformer(n_lobby_modes, n_modes, n_stages, n_weapons, n_ranks, size='normal', predict=True)
    model.load_state_dict(torch.load(weights, map_location=device))
    model = model.eval().to(device)
    
    # test
    outputs = []
    probs = []
    ys = []
    for sdata, y in tqdm(dataloader):
        sdata = sdata_to(sdata, device)
        y = y.to(device)
        
        out = model(sdata)
        
        batch_prob = [list(p) for p in out.detach().cpu().numpy()]
        batch_output = list(out.argmax(axis=-1).cpu().numpy())
        batch_y = list(y.cpu().numpy())
        
        outputs += batch_output
        probs += batch_prob
        ys += batch_y
        
    accuracy = accuracy_score(outputs, ys)
    precision = precision_score(outputs, ys)
    recall = recall_score(outputs, ys)
    f1 = f1_score(outputs, ys)
    
    df = []
    for out, y, prob in zip(outputs, ys, probs):
        df.append({
            'pred': int(out),
            'y': int(y),
            'prob': max(prob),
        })
    os.makedirs(str(Path(outfile).absolute().parent), exist_ok=True)
    pd.DataFrame(df).to_csv(outfile, index=False, header=True)
    
    print('Data Size: {}'.format(len(dataloader)))
    print('Accuracy:  {:.3f}'.format(accuracy))
    print('Precision: {:.3f}'.format(precision))
    print('Recall:    {:.3f}'.format(recall))
    print('F1:        {:.3f}'.format(f1))
            
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--ds-path', type=str, help='dataset path')
    parser.add_argument('--weights', type=str, help='weights path')
    parser.add_argument('--outfile', type=str, default='output.csv', help='output file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = build_parser()
    
    run_test(args.ds_path, args.weights, args.outfile)
    
