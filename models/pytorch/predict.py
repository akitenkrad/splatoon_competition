import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .datasets import SplatoonDataset, sdata_to
from .model import SimpleTransformer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def run_predict(ds_path: str, weights: str, outfile: str):
 
    assert Path(outfile).suffix == '.csv'
    os.makedirs(str(Path(outfile).absolute().parent), exist_ok=True)
       
    epochs = 1
    batch_size = 128
    
    # device
    device = torch.device('cpu')
    
    # load dataset
    dataset = SplatoonDataset(str(Path(ds_path).absolute()), phase='test')
    
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
    for sdata, _ in tqdm(dataloader):
        sdata = sdata_to(sdata, device)
        
        out = model(sdata)
        batch_output = out.argmax(axis=-1).cpu().numpy()
        for idx, output in zip(sdata.idx, batch_output):
            outputs.append({'id': int(idx.cpu().numpy()), 'y': int(output)}) 
 
    pd.DataFrame(outputs).to_csv(outfile, index=False, header=True)
    
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--ds-path', type=str, help='dataset path')
    parser.add_argument('--weights', type=str, help='weights path')
    parser.add_argument('--outfile', type=str, help='output csv file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = build_parser()
    
    run_predict(args.ds_path, args.weights, args.outfile)
    
