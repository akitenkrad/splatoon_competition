import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from datetime import datetime
from collections import namedtuple

SData = namedtuple('SData', ('idx', 'period', 'game_ver', 'lobby_mode', 'lobby', 'mode', 'stage',
                             'a1_weapon', 'a1_rank', 'a1_level', 'a2_weapon', 'a2_rank', 'a2_level',
                             'a3_weapon', 'a3_rank', 'a3_level', 'a4_weapon', 'a4_rank', 'a4_level',
                             'b1_weapon', 'b1_rank', 'b1_level', 'b2_weapon', 'b2_rank', 'b2_level',
                             'b3_weapon', 'b3_rank', 'b3_level', 'b4_weapon', 'b4_rank', 'b4_level'))

XData = namedtuple('XData', ('lobby_mode', 'mode', 'stage', 
                             'player_a1', 'player_a2', 'player_a3', 'player_a4',
                             'player_b1', 'player_b2', 'player_b3', 'player_b4'))

def sdata_to(sdata, device):
    new_sdata = SData(sdata.idx.to(device), sdata.period, sdata.game_ver, sdata.lobby_mode.to(device),
                      sdata.lobby, sdata.mode.to(device), sdata.stage.to(device),
                      sdata.a1_weapon.to(device), sdata.a1_rank.to(device), sdata.a1_level.to(device),
                      sdata.a2_weapon.to(device), sdata.a2_rank.to(device), sdata.a2_level.to(device),
                      sdata.a3_weapon.to(device), sdata.a3_rank.to(device), sdata.a3_level.to(device),
                      sdata.a4_weapon.to(device), sdata.a4_rank.to(device), sdata.a4_level.to(device),
                      sdata.b1_weapon.to(device), sdata.b1_rank.to(device), sdata.b1_level.to(device),
                      sdata.b2_weapon.to(device), sdata.b2_rank.to(device), sdata.b2_level.to(device),
                      sdata.b3_weapon.to(device), sdata.b3_rank.to(device), sdata.b3_level.to(device),
                      sdata.b4_weapon.to(device), sdata.b4_rank.to(device), sdata.b4_level.to(device))
    return new_sdata
    
class Vocabulary(object):
    def __init__(self, sos_token=None, eos_token=None):
        self.__tokens = []
        if sos_token is not None:
            self.__tokens.append(sos_token)
        if eos_token is not None:
            self.__tokens.append(eos_token)
        
    def __len__(self):
        return len(self.__tokens)
    
    def add_token(self, token):
        if token not in self.__tokens:
            self.__tokens.append(token)
            
    def tok2idx(self, token):
        return self.__tokens.index(token)
    
    def idx2tok(self, index):
        return self.__tokens[index]
    
class SplatoonDataset(torch.utils.data.IterableDataset):
    
    def __init__(self, filename):
        self.filename = filename
        self.__build_vocab()
        
    def __len__(self):
        return self.total
    
    def preprocess(self, text, with_vocab=False):
        # 'id,period,game-ver,lobby-mode,lobby,mode,stage,A1-weapon,A1-rank,A1-level,A2-weapon,A2-rank,A2-level,A3-weapon,A3-rank,A3-level,A4-weapon,A4-rank,A4-level,B1-weapon,B1-rank,B1-level,B2-weapon,B2-rank,B2-level,B3-weapon,B3-rank,B3-level,B4-weapon,B4-rank,B4-level,y
        items = [t for t in text.lower().strip().split(',')]
        idx = int(items[0])
        period, game_ver, lobby_mode, lobby, mode, stage = [str(i) for i in items[1:7]]
        a1_w, a1_r, a1_l = str(items[7]), str(items[8]), int(float(items[9]) if items[9] != '' else 0)
        a2_w, a2_r, a2_l = str(items[10]), str(items[11]), int(float(items[12]) if items[12] != '' else 0)
        a3_w, a3_r, a3_l = str(items[13]), str(items[14]), int(float(items[15]) if items[15] != '' else 0)
        a4_w, a4_r, a4_l = str(items[16]), str(items[17]), int(float(items[18]) if items[18] != '' else 0)
        b1_w, b1_r, b1_l = str(items[19]), str(items[20]), int(float(items[21]) if items[21] != '' else 0)
        b2_w, b2_r, b2_l = str(items[22]), str(items[23]), int(float(items[24]) if items[24] != '' else 0)
        b3_w, b3_r, b3_l = str(items[25]), str(items[26]), int(float(items[27]) if items[27] != '' else 0)
        b4_w, b4_r, b4_l = str(items[28]), str(items[29]), int(float(items[30]) if items[30] != '' else 0)
        y = int(items[31])
        
        if with_vocab:
            sdata = SData(idx, period, game_ver,
                          self.lobby_mode_vocab.tok2idx(lobby_mode), lobby, self.mode_vocab.tok2idx(mode), self.stage_vocab.tok2idx(stage), 
                          self.weapon_vocab.tok2idx(a1_w), self.rank_vocab.tok2idx(a1_r), np.float32((a1_l - self.level_mean) / self.level_std),
                          self.weapon_vocab.tok2idx(a2_w), self.rank_vocab.tok2idx(a2_r), np.float32((a2_l - self.level_mean) / self.level_std),
                          self.weapon_vocab.tok2idx(a3_w), self.rank_vocab.tok2idx(a3_r), np.float32((a3_l - self.level_mean) / self.level_std),
                          self.weapon_vocab.tok2idx(a4_w), self.rank_vocab.tok2idx(a4_r), np.float32((a4_l - self.level_mean) / self.level_std),
                          self.weapon_vocab.tok2idx(b1_w), self.rank_vocab.tok2idx(b1_r), np.float32((b1_l - self.level_mean) / self.level_std),
                          self.weapon_vocab.tok2idx(b2_w), self.rank_vocab.tok2idx(b2_r), np.float32((b2_l - self.level_mean) / self.level_std),
                          self.weapon_vocab.tok2idx(b3_w), self.rank_vocab.tok2idx(b3_r), np.float32((b3_l - self.level_mean) / self.level_std),
                          self.weapon_vocab.tok2idx(b4_w), self.rank_vocab.tok2idx(b4_r), np.float32((b4_l - self.level_mean) / self.level_std))
        else:
            sdata = SData(idx, period, game_ver, lobby_mode, lobby, mode, stage, 
                          a1_w, a1_r, a1_l, a2_w, a2_r, a2_l, a3_w, a3_r, a3_l, a4_w, a4_r, a4_l,
                          b1_w, b1_r, b1_l, b2_w, b2_r, b2_l, b3_w, b3_r, b3_l, b4_w, b4_r, b4_l)
        return sdata, y
    
    def __build_vocab(self):
        self.lobby_mode_vocab = Vocabulary()
        self.mode_vocab = Vocabulary()
        self.stage_vocab = Vocabulary()
        self.weapon_vocab = Vocabulary()
        self.rank_vocab = Vocabulary()
        self.level_mean = 0
        self.level_std = 0
        self.total = 0
        
        with open(self.filename) as f_it:
            
            next(f_it)  # skip header
            # count all data
            for _ in f_it:
                self.total += 1

            # seek to the beginning of the file
            f_it.seek(0)
            next(f_it)  # skip header
            
            levels = []
        
            for line in tqdm(f_it, desc='building vocabulary', total=self.total):
                sdata, _ = self.preprocess(line)

                self.lobby_mode_vocab.add_token(sdata.lobby_mode)
                self.mode_vocab.add_token(sdata.mode)
                self.stage_vocab.add_token(sdata.stage)

                self.weapon_vocab.add_token(str(sdata.a1_weapon))
                self.weapon_vocab.add_token(str(sdata.a2_weapon))
                self.weapon_vocab.add_token(str(sdata.a3_weapon))
                self.weapon_vocab.add_token(str(sdata.a4_weapon))
                self.weapon_vocab.add_token(str(sdata.b1_weapon))
                self.weapon_vocab.add_token(str(sdata.b2_weapon))
                self.weapon_vocab.add_token(str(sdata.b3_weapon))
                self.weapon_vocab.add_token(str(sdata.b4_weapon))

                self.rank_vocab.add_token(str(sdata.a1_rank))
                self.rank_vocab.add_token(str(sdata.a2_rank))
                self.rank_vocab.add_token(str(sdata.a3_rank))
                self.rank_vocab.add_token(str(sdata.a4_rank))
                self.rank_vocab.add_token(str(sdata.b1_rank))
                self.rank_vocab.add_token(str(sdata.b2_rank))
                self.rank_vocab.add_token(str(sdata.b3_rank))
                self.rank_vocab.add_token(str(sdata.b4_rank))
                
                levels.append(sdata.a1_level)
                levels.append(sdata.a2_level)
                levels.append(sdata.a3_level)
                levels.append(sdata.a4_level)
                levels.append(sdata.b1_level)
                levels.append(sdata.b2_level)
                levels.append(sdata.b3_level)
                levels.append(sdata.b4_level)
            
            self.level_mean = np.mean(levels)
            self.level_std = np.std(levels)
            
    def __line_mapper(self, line):
        sdata, y = self.preprocess(line ,with_vocab=True)
        
        return sdata, y
    
    def __iter__(self):
        f_itr = open(self.filename)
        
        # skip headers
        next(f_itr)
        
        mapped_itr = map(self.__line_mapper, f_itr)
        
        return mapped_itr
