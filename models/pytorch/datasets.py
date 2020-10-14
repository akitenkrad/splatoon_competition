import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from collections import namedtuple
from sklearn.model_selection import train_test_split

class PlayerData(object):
    def __init__(self, weapon, rank, level, sub_weapon, special_weapon, weapon_range, weapon_power, weapon_rounds_per, weapon_iine):
        self.weapon = weapon
        self.rank = rank
        self.level = level
        self.sub_weapon = sub_weapon
        self.special_weapon = special_weapon
        self.weapon_range = weapon_range
        self.weapon_power = weapon_power
        self.weapon_rounds_per = weapon_rounds_per
        self.weapon_iine = weapon_iine
    
    def apply_vocab(self, weapon_vocab, rank_vocab, sub_weapon_vocab, special_weapon_vocab):
        self.weapon = weapon_vocab.tok2idx(self.weapon)
        self.rank = rank_vocab.tok2idx(self.rank)
        self.sub_weapon = sub_weapon_vocab.tok2idx(self.sub_weapon)
        self.special_weapon = special_weapon_vocab.tok2idx(self.special_weapon)

    def to_dict(self):
        data = {
            'weapon': self.weapon,
            'rank': self.rank,
            'level': self.level,
            'sub_weapon': self.sub_weapon,
            'special_weapon': self.special_weapon,
            'weapon_range': self.weapon_range,
            'weapon_power': self.weapon_power,
            'weapon_rounds_per': self.weapon_rounds_per,
            'weapon_iine': self.weapon_iine,
            }
        
        return data

class SplatoonData(object):
    def __init__(self, line, phase, weapon_data: dict={}):
        self.line = line
        items = [t for t in line.lower().strip().split(',')]
        self.idx = int(items[0])
        self.period, self.game_ver, self.lobby_mode, self.lobby, self.mode, self.stage = [str(i) for i in items[1:7]]
        self.y = int(items[31]) if phase == 'train' else 0
        
        # build player
        self.a_players = []
        self.b_players = []
        player_map = {'a1': 7, 'a2': 10, 'a3': 13, 'a4': 16, 'b1': 19, 'b2': 22, 'b3': 25, 'b4': 28}
        for player, pidx in player_map.items():
            weapon, rank, level = str(items[pidx]), str(items[pidx+1]), int(float(items[pidx+2]) if items[pidx+2] != '' else 0)
            sub_weapon, special_weapon, weapon_range, weapon_power, weapon_rounds_per, weapon_iine = weapon_data[weapon]
            
            if player[0] == 'a':
                self.a_players.append(PlayerData(weapon, rank, level, sub_weapon, special_weapon, weapon_range, weapon_power, weapon_rounds_per, weapon_iine))
            elif player[0] == 'b':
                self.b_players.append(PlayerData(weapon, rank, level, sub_weapon, special_weapon, weapon_range, weapon_power, weapon_rounds_per, weapon_iine))


    def apply_vocab(self, lobby_mode_vocab, mode_vocab, stage_vocab, weapon_vocab, rank_vocab, sub_weapon_vocab, special_weapon_vocab):
        self.lobby_mode = lobby_mode_vocab.tok2idx(self.lobby_mode)
        self.mode = mode_vocab.tok2idx(self.mode)
        self.stage = stage_vocab.tok2idx(self.stage)
        for player in self.a_players:
            player.apply_vocab(weapon_vocab, rank_vocab, sub_weapon_vocab, special_weapon_vocab)
        for player in self.b_players:
            player.apply_vocab(weapon_vocab, rank_vocab, sub_weapon_vocab, special_weapon_vocab)

    def to_dict(self):
        data = {
            'idx': self.idx,
            'period': self.period,
            'game_ver': self.game_ver,
            'lobby_mode': self.lobby_mode,
            'lobby': self.lobby,
            'mode': self.mode,
            'stage': self.stage,
            'a_players': [p.to_dict() for p in self.a_players],
            'b_players': [p.to_dict() for p in self.b_players],
            'y': self.y,
        }
        return data


class Vocabulary(object):
    def __init__(self, sos_token=None, eos_token=None):
        self.__tokens = []
        self.sos_token = sos_token
        self.eos_token = eos_token
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
    
    def copy(self):
        v = Vocabulary(self.sos_token, self.eos_token)
        for token in self.__tokens:
            v.add_token(token)
        return v


class SplatoonDataset(torch.utils.data.Dataset):
    
    def __init__(self, filename, weapon_data_path, phase='train', lines=[]):
        '''
        data format:
            {
                'idx': ID,
                'period': PERIOD,
                'game_ver': GAME VERSION,
                'lobby_mode': LOBBY MODE
                'lobby': LOBBY,
                'mode': MODE,
                'stage': STAGE
                'a_players': [{
                    'weapon': WEAPON,
                    'rank': RANK,
                    'level': LEVEL,
                    'sub_weapon': SUB_WEAPON,
                    'special_weapon': SPECIAL_WEAPON,
                    'weapon_range': WEAPON_RANGE,
                    'weapon_power': WEAPON_POWER,
                    'weapon_rounds_per': WEAPON_ROUNDS_PER,
                    'weapon_iine': WEAPON_IINE,
                    }],
                'b_players': [{
                    'weapon': WEAPON,
                    'rank': RANK,
                    'level': LEVEL,
                    'sub_weapon': SUB_WEAPON,
                    'special_weapon': SPECIAL_WEAPON,
                    'weapon_range': WEAPON_RANGE,
                    'weapon_power': WEAPON_POWER,
                    'weapon_rounds_per': WEAPON_ROUNDS_PER,
                    'weapon_iine': WEAPON_IINE,
                    }],
                'y': Y
            }
            
        Args:
            filename (str): dataset file name (train_data.csv)
            weapon_data_path (str): weapon external data (main_weapon_data.csv)
            phase (str): 'train' or 'test'
        '''
        assert phase in ['train', 'test']
        
        self.phase = phase
        self.filename = filename
        
        # load weapon data
        self.weapon_data_path = weapon_data_path
        self.weapon_data = self.__load_weapon_data(self.weapon_data_path)
        
        if lines:
            self.lines = lines
        else:
            self.lines = [l.strip() for idx, l in enumerate(open(filename)) if idx > 0]
            self.__build_vocab()
        
    def __len__(self):
        return self.total
    
    def preprocess(self, text):
        # 'id,period,game-ver,lobby-mode,lobby,mode,stage,A1-weapon,A1-rank,A1-level,A2-weapon,A2-rank,A2-level,A3-weapon,A3-rank,A3-level,A4-weapon,A4-rank,A4-level,B1-weapon,B1-rank,B1-level,B2-weapon,B2-rank,B2-level,B3-weapon,B3-rank,B3-level,B4-weapon,B4-rank,B4-level,y
        data = SplatoonData(text, self.phase, self.weapon_data)
        return data
    
    def apply_vocab(self, data):
        data.apply_vocab(self.lobby_mode_vocab, self.mode_vocab, self.stage_vocab, self.weapon_vocab, self.rank_vocab, self.sub_weapon_vocab, self.special_weapon_vocab)
        return data
        
    @property
    def total(self):
        return len(self.lines)
    
    def __load_weapon_data(self, weapon_data_path):
        weapon_data = {'': ('None', 'None', 0, 0, 0, 0)}
        for idx, row in pd.read_csv(weapon_data_path).iterrows():
            weapon_data[row.key] = (row.sub_weapon, row.special, row.range, row.power, row.rounds_per, row.iine)
        return weapon_data
        
    def __build_vocab(self):
        self.lobby_mode_vocab = Vocabulary()
        self.mode_vocab = Vocabulary()
        self.stage_vocab = Vocabulary()
        self.weapon_vocab = Vocabulary()
        self.rank_vocab = Vocabulary()
        self.sub_weapon_vocab = Vocabulary()
        self.special_weapon_vocab = Vocabulary()
        
        # build dataset vocabulary
        for line in tqdm(self.lines, desc='Building Vocabulary'):
        
            data = self.preprocess(line)

            self.lobby_mode_vocab.add_token(data.lobby_mode)
            self.mode_vocab.add_token(data.mode)
            self.stage_vocab.add_token(data.stage)

            for player in data.a_players + data.b_players:
                self.weapon_vocab.add_token(str(player.weapon))
                self.rank_vocab.add_token(str(player.rank))
                self.sub_weapon_vocab.add_token(str(player.sub_weapon))
                self.special_weapon_vocab.add_token(str(player.special_weapon))
                
    def __getitem__(self, index):
        target_line = self.lines[index]
        data = self.preprocess(target_line)
        data = self.apply_vocab(data)
        
        return data.to_dict()

    def copy_vocab(self, dataset):
        self.lobby_mode_vocab = dataset.lobby_mode_vocab.copy()
        self.mode_vocab = dataset.mode_vocab.copy()
        self.rank_vocab = dataset.rank_vocab.copy()
        self.stage_vocab = dataset.stage_vocab.copy()
        self.weapon_vocab = dataset.weapon_vocab.copy()
        self.sub_weapon_vocab = dataset.sub_weapon_vocab.copy()
        self.special_weapon_vocab = dataset.special_weapon_vocab.copy()
        
    def train_test_split(self, test_size=0.1):
        train_lines, test_lines = train_test_split(self.lines, test_size=test_size)
        train_ds = SplatoonDataset(self.filename, phase=self.phase, weapon_data_path=self.weapon_data_path, lines=train_lines)
        test_ds = SplatoonDataset(self.filename, phase=self.phase, weapon_data_path=self.weapon_data_path, lines=test_lines)
        
        train_ds.copy_vocab(self)
        test_ds.copy_vocab(self)
        
        return train_ds, test_ds
    
    @classmethod
    def to(cls, data, device):
        new_data = {
            'lobby_mode': data['lobby_mode'].to(device),
            'mode': data['mode'].to(device),
            'stage': data['stage'].to(device),
            'a_players': [{key: val.to(device) for key, val in player.items()} for player in data['a_players']],
            'b_players': [{key: val.to(device) for key, val in player.items()} for player in data['b_players']],
            'y': data['y'].to(device),
        }
        return new_data