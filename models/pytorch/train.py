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
from .datasets import SplatoonDataset
from .model import SimpleTransformer

class History(object):
    def __init__(self):
        self.history = SummaryWriter(log_dir='./log')
        self.best_accuracy = -1.0
        self.is_best = False
        self.epoch = 0
        self.loss = -1
        self.train_accuracy, self.train_precision, self.train_recall, self.train_f1 = 0.0, 0.0, 0.0, 0.0
        self.test_accuracy, self.test_precision, self.test_recall, self.test_f1 = 0.0, 0.0, 0.0, 0.0
    
    def add_train_value(self, epoch, outputs, ys, loss):
        self.epoch = epoch
        self.loss = loss
        self.train_accuracy = accuracy_score(ys, outputs)
        self.train_precision = precision_score(ys, outputs)
        self.train_recall = recall_score(ys, outputs)
        self.train_f1 = f1_score(ys, outputs)
        
        self.history.add_scalar('loss', self.loss, epoch)
        self.history.add_scalar('train_accuracy', self.train_accuracy, epoch)
        self.history.add_scalar('train_precision', self.train_precision, epoch)
        self.history.add_scalar('train_recall', self.train_recall, epoch)
        self.history.add_scalar('train_f1', self.train_f1, epoch)
        
        if self.best_accuracy < self.train_accuracy:
            self.best_accuracy = self.train_accuracy
            self.is_best = True
        else:
            self.is_best = False
        
    def add_test_value(self, epoch, outputs, ys):
        self.test_accuracy = accuracy_score(ys, outputs)
        self.test_precision = precision_score(ys, outputs)
        self.test_recall = recall_score(ys, outputs)
        self.test_f1 = f1_score(ys, outputs)
        
        self.history.add_scalar('test_accuracy', self.test_accuracy, epoch)
        self.history.add_scalar('test_precision', self.test_precision, epoch)
        self.history.add_scalar('test_recall', self.test_recall, epoch)
        self.history.add_scalar('test_f1', self.test_f1, epoch)
    
    def description(self):
        desc = 'Epoch:{} Loss:{:.6f} Tran-Acc:{:.6f} Train-F1:{:.6f} Test-Acc:{:.6f} Test-F1:{:.6f}'.format(
            self.epoch, self.loss, self.train_accuracy, self.train_f1, self.test_accuracy, self.test_f1)
        return desc

def run_train(ds_path: str, weapon_ds_path:str, epochs: int, batch_size: int, test_size: float=0.1, checkpoint: str=''):
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load dataset
    dataset = SplatoonDataset(str(Path(ds_path).absolute()), str(Path(weapon_ds_path).absolute()), phase='train')
    train_ds, test_ds = dataset.train_test_split(test_size=test_size)
    
    # read meta counts
    n_lobby_modes = len(dataset.lobby_mode_vocab)
    n_modes = len(dataset.mode_vocab)
    n_weapons = len(dataset.weapon_vocab)
    n_sub_weapons = len(dataset.sub_weapon_vocab)
    n_special_weapons = len(dataset.special_weapon_vocab)
    n_ranks = len(dataset.rank_vocab)
    n_stages = len(dataset.stage_vocab)
    
    # split dataset
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # prepare model, optimizer, criterion
    model = SimpleTransformer(n_lobby_modes, n_modes, n_stages, n_weapons, n_sub_weapons, n_special_weapons, n_ranks)
    optimizer = optim.Adam(model.parameters(), lr=1e-8)
    criterion = nn.CrossEntropyLoss()
    
    if checkpoint:
      model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
      optimizer.load_state_dict(torch.load(checkpoint)['optimizer_state_dict'])
    
    model = model.train().to(device)
    
    # history
    history = History()
    
    # save dir
    save_dir = Path('./weights')
    os.makedirs(str(save_dir), exist_ok=True)
    
    # === run epoch ========================
    with tqdm(total=epochs, leave=True) as epoch_pbar:
        for epoch in range(epochs):

            optimizer.zero_grad()
            
            # === train ========================
            outputs, ys, losses = np.array([]), np.array([]), []
            with tqdm(total=len(train_dl), leave=None) as train_pbar:
                for data in train_dl:
                    data = SplatoonDataset.to(data, device)
                    y = data['y']

                    out = model(data)

                    # update parameters
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()

                    batch_output = out.argmax(axis=-1).cpu().numpy()
                    batch_y = y.cpu().numpy()
                    batch_loss = float(loss.detach().cpu().numpy())

                    train_pbar.update(1)
                    train_pbar.set_description('Epoch:{} Loss:{:.6f} Acc:{:.6f} F1:{:.6f}'.format(
                        epoch + 1,
                        batch_loss, 
                        accuracy_score(batch_output, batch_y),
                        f1_score(batch_output, batch_y)))

                    outputs = np.hstack([outputs, batch_output])
                    ys = np.hstack([ys, batch_y])
                    losses.append(batch_loss)

                history.add_train_value(epoch + 1, outputs, ys, np.mean(losses))

            # === test ========================
            outputs, ys = np.array([]), np.array([])
            with torch.no_grad():
                with tqdm(total=len(test_dl), leave=None) as test_pbar:
                    for data in test_dl:
                        data = SplatoonDataset.to(data, device)
                        y = data['y']

                        out = model(data)

                        batch_output = out.argmax(axis=-1).cpu().numpy()
                        batch_y = y.cpu().numpy()

                        test_pbar.update(1)
                        test_pbar.set_description('Epoch:{} Loss:{:.6f} Acc:{:.6f} F1:{:.6f}'.format(
                            epoch + 1,
                            batch_loss, 
                            accuracy_score(batch_output, batch_y),
                            f1_score(batch_output, batch_y)))

                        outputs = np.hstack([outputs, batch_output])
                        ys = np.hstack([ys, batch_y])

                    history.add_test_value(epoch + 1, outputs, ys)

            # === save weights ==================
            epoch_pbar.update(1)
            epoch_pbar.set_description(history.description())

            if history.is_best:
                best_path = save_dir / 'best.pth'
                torch.save(model.state_dict(), best_path)

                checkpoint = save_dir / 'checkpoint_best.pth'
                torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': history.loss,
                  }, checkpoint)

            if (epoch + 1) % 20 == 0:
                last_path = save_dir / 'last_at_{}.pth'.format(epoch + 1)
                torch.save(model.state_dict(), last_path)

            # save checkpoint
            checkpoint = save_dir / 'checkpoint.pth'
            torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': history.loss,
              }, checkpoint)
 
    
    
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--ds-path', type=str, help='dataset path')
    parser.add_argument('--weapon-ds-path', type=str, help='weapon dataset path')
    parser.add_argument('--epochs', type=int, default=100, help='epochs. default=100')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size. default=8')
    parser.add_argument('--test-size', type=float, default=0.1, help='test size')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint path')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = build_parser()
    
    run_train(args.ds_path, args.weapon_ds_path, args.epochs, args.batch_size, args.test_size, args.checkpoint)
    
