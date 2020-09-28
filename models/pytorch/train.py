import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .datasets import SplatoonDataset, sdata_to
from .model import SimpleTransformer

def run_train(ds_path: str, epochs: int, batch_size: int):
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load dataset
    dataset = SplatoonDataset(str(Path(ds_path).absolute()))
    
    # read meta counts
    n_lobby_modes = len(dataset.lobby_mode_vocab)
    n_modes = len(dataset.mode_vocab)
    n_weapons = len(dataset.weapon_vocab)
    n_ranks = len(dataset.rank_vocab)
    n_stages = len(dataset.stage_vocab)
    
    # split dataset
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # prepare model
    model = SimpleTransformer(n_lobby_modes, n_modes, n_stages, n_weapons, n_ranks)
    model = model.train().to(device)
    
    # prepare optimizer, criterion
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # history
    history = SummaryWriter(log_dir='./logs')
    
    # best_acc
    best_acc = 0
    
    # save dir
    save_dir = Path('./weights')
    os.makedirs(str(save_dir), exist_ok=True)
    
    # run
    for epoch in tqdm(range(epochs)):
        
        is_best = False
        
        # train
        outputs = np.array([])
        ys = np.array([])
        losses = []
        for sdata, y in tqdm(dataloader, desc='Epoch: {}'.format(epoch + 1), leave=False):
            sdata = sdata_to(sdata, device)
            y = y.to(device)
            
            out = model(sdata)
            
            # update parameters
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
         
            outputs = np.hstack([outputs, out.argmax(axis=-1).cpu().numpy()])
            ys = np.hstack([ys, y.cpu().numpy()])
            losses.append(float(loss.detach().cpu().numpy()))
            
        history.add_scalar('loss', np.mean(losses), epoch + 1)
        history.add_scalar('train_accuracy', accuracy_score(outputs, ys), epoch + 1)
        history.add_scalar('train_precision', precision_score(outputs, ys), epoch + 1)
        history.add_scalar('train_recall', recall_score(outputs, ys), epoch + 1)
        history.add_scalar('train_f1', f1_score(outputs, ys), epoch + 1)
        
        cur_acc = accuracy_score(outputs, ys)
        if best_acc < cur_acc:
            best_acc = cur_acc
            is_best = True
            
        if is_best:
            best_path = save_dir / 'best.pth'
            torch.save(model.state_dict(), best_path)
        
        if (epoch + 1) % 10 == 0:
            last_path = save_dir / 'last.pth'
            torch.save(model.state_dict(), last_path)
            
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--ds-path', type=str, help='dataset path')
    parser.add_argument('--epochs', type=int, default=100, help='epochs. default=100')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size. default=8')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = build_parser()
    
    run_train(args.ds_path, args.epochs, args.batch_size)
    
