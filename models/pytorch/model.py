import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dateutil.parser import parse as date_parse

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x):
        embedded = self.embed(x)
        return embedded
    
class PlayerEmbedding(nn.Module):
    def __init__(self,
                 weapon_embed: nn.Embedding,
                 rank_embed: nn.Embedding,
                 sub_weapon_embed: nn.Embedding,
                 special_weapon_embed: nn.Embedding):
        super().__init__()
        self.weapon_embed = weapon_embed
        self.rank_embed = rank_embed
        self.sub_weapon_embed = sub_weapon_embed
        self.special_weapon_embed = special_weapon_embed
    
    @property
    def n_dim(self):
        n_level, n_weapon_range, n_weapon_power, n_weapon_rounds_per, n_weapon_iine = 1, 1, 1, 1, 1
        d = (self.weapon_embed.embedding_dim + 
             self.rank_embed.embedding_dim +
             self.sub_weapon_embed.embedding_dim +
             self.special_weapon_embed.embedding_dim +
             n_level + n_weapon_range + n_weapon_power + n_weapon_rounds_per + n_weapon_iine)
        return d
            
            
    def forward(self, weapon, rank, level, sub_weapon, special_weapon, weapon_range, weapon_power, weapon_rounds_per, weapon_iine):
        weapon_embedded = self.weapon_embed(weapon)
        rank_embedded = self.rank_embed(rank)
        sub_weapon_embedded = self.sub_weapon_embed(sub_weapon)
        special_weapon_embedded = self.special_weapon_embed(special_weapon)
        
        feat = torch.cat([
            weapon_embedded,
            rank_embedded,
            sub_weapon_embedded,
            special_weapon_embedded,
            level.unsqueeze(-1),
            weapon_range.unsqueeze(-1),
            weapon_power.unsqueeze(-1),
            weapon_rounds_per.unsqueeze(-1),
            weapon_iine.unsqueeze(-1),
            ], axis=1)
        
        return feat

class Attention(nn.Module):
    def __init__(self, dim: int=300):
        super().__init__()
        
        self.q_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        
        self.out = nn.Linear(dim, dim)
        
        self.d_k = dim
        
    def forward(self, q, k, v):
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        
        weights = torch.matmul(q.unsqueeze(-1), k.unsqueeze(1)) / math.sqrt(self.d_k)
        weights = weights.reshape(-1, self.d_k, self.d_k)
        
        normalized_weights = F.softmax(weights, dim=-1)
        
        output = torch.matmul(normalized_weights, v.unsqueeze(-1)).squeeze()
        output = self.out(output)
        
        return output, normalized_weights

class FeedForward(nn.Module):
    def __init__(self, dim, d_ff=1024, dropout=0.1):
        super().__init__()
        
        self.linear_1 = nn.Linear(dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, dim)
 
    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x

class TransformerCell(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        
        self.attn = Attention(dim)
        self.ff = FeedForward(dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x):
        output, normalized_weights = self.attn(x, x, x)
        x2 = x + self.dropout_1(output)
        output = x2 + self.dropout_2(self.ff(x2))
        
        return output, normalized_weights

class TransformerBlock(nn.Module):
    def __init__(self, player_dim: int, lobby_mode_dim: int, mode_dim: int, stage_dim: int, dropout=0.1):
        super().__init__()
        
        self.lobby_mode_cell = TransformerCell(lobby_mode_dim)
        self.mode_cell = TransformerCell(mode_dim)
        self.stage_cell = TransformerCell(stage_dim)
        
        self.a_team_cell = TransformerCell(player_dim)
        self.b_team_cell = TransformerCell(player_dim)
 
    def forward(self, x):
        lobby_mode, lobby_mode_attn = self.lobby_mode_cell(x['lobby_mode'])
        mode, mode_attn = self.mode_cell(x['mode'])
        stage, stage_attn = self.stage_cell(x['stage'])
 
        a_team, a_team_attn = self.a_team_cell(x['a_team'])
        b_team, b_team_attn = self.b_team_cell(x['b_team'])
        
        output = {
            'lobby_mode': lobby_mode,
            'mode': mode,
            'stage': stage,
            'a_team': a_team,
            'b_team': b_team,
        }
               
        attention_weights = {
            'lobby_mode': lobby_mode_attn,
            'mode': mode_attn,
            'stage': stage_attn,
            'a_team': a_team,
            'b_team': b_team,
        }
        
        return output, attention_weights
    
class SimpleTransformer(nn.Module):
    def __init__(self,
                 n_lobby_modes: int,
                 n_modes: int,
                 n_stages: int,
                 n_weapons: int,
                 n_sub_weapons: int,
                 n_special_weapons: int,
                 n_ranks: int,
                 size='small',
                 predict=False):
        super().__init__()
        assert size in ['small', 'normal', 'large']
        
        self.size = size
        self.predict = predict
        self.weapon_embed = Embedding(n_weapons, n_weapons)
        self.sub_weapon_embed = Embedding(n_sub_weapons, n_sub_weapons)
        self.special_weapon_embed = Embedding(n_special_weapons, n_special_weapons)
        self.rank_embed = Embedding(n_ranks, n_ranks)
        self.lobby_mode_embed = Embedding(n_lobby_modes, n_lobby_modes)
        self.mode_embed = Embedding(n_modes, n_modes)
        self.stage_embed = Embedding(n_stages, n_stages)
        
        self.a_players = [
            PlayerEmbedding(self.weapon_embed, self.rank_embed, self.sub_weapon_embed, self.special_weapon_embed),
            PlayerEmbedding(self.weapon_embed, self.rank_embed, self.sub_weapon_embed, self.special_weapon_embed),
            PlayerEmbedding(self.weapon_embed, self.rank_embed, self.sub_weapon_embed, self.special_weapon_embed),
            PlayerEmbedding(self.weapon_embed, self.rank_embed, self.sub_weapon_embed, self.special_weapon_embed),
            ]
        
        self.b_players = [
            PlayerEmbedding(self.weapon_embed, self.rank_embed, self.sub_weapon_embed, self.special_weapon_embed),
            PlayerEmbedding(self.weapon_embed, self.rank_embed, self.sub_weapon_embed, self.special_weapon_embed),
            PlayerEmbedding(self.weapon_embed, self.rank_embed, self.sub_weapon_embed, self.special_weapon_embed),
            PlayerEmbedding(self.weapon_embed, self.rank_embed, self.sub_weapon_embed, self.special_weapon_embed),
        ]
        
        self.player_dim = self.a_players[0].n_dim
        self.lobby_mode_dim = self.lobby_mode_embed.embedding_dim
        self.mode_dim = self.mode_embed.embedding_dim
        self.stage_dim = self.stage_embed.embedding_dim
        
        self.transformer_blocks = [
            TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim),
            TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim),
            TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim),
            TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim),
            TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim),
            TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim),
        ]
        
        if self.size == 'normal':
            self.transformer_blocks += [
                TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim),
                TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim),
                TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim),
                TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim),
                TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim),
                TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim),
            ]
        
        self.out = nn.Linear(self.lobby_mode_dim + self.mode_dim + self.stage_dim + self.player_dim * 2, 2)
        self.out_softmax = torch.nn.Softmax(dim=1)
        
        self.attention_weights = []
        
    def forward(self, x):
        '''
        Args:
            x: {
                'idx': ID,
                'period': PERIOD,
                'game_ver': GAME VERSION,
                'lobby_mode': LOBBY MODE
                'lobby': LOBBY,
                'mode': MODE,
                'stage': STAGE
                'a_players': [{
                    'weapon': WEAPON, 'rank': RANK, 'level': LEVEL, 'sub_weapon': SUB_WEAPON, 'special_weapon': SPECIAL_WEAPON,
                    'weapon_range': WEAPON_RANGE, 'weapon_power': WEAPON_POWER, 'weapon_rounds_per': WEAPON_ROUNDS_PER, 'weapon_iine': WEAPON_IINE,
                    }],
                'b_players': [{
                    'weapon': WEAPON, 'rank': RANK, 'level': LEVEL, 'sub_weapon': SUB_WEAPON, 'special_weapon': SPECIAL_WEAPON,
                    'weapon_range': WEAPON_RANGE, 'weapon_power': WEAPON_POWER, 'weapon_rounds_per': WEAPON_ROUNDS_PER, 'weapon_iine': WEAPON_IINE,
                    }],
                'y': Y
            }       
        '''
        self.attention_weights = []

        lobby_mode = self.lobby_mode_embed(x['lobby_mode'])
        mode = self.mode_embed(x['mode'])
        stage = self.stage_embed(x['stage'])
        
        # weapon, rank, level, sub_weapon, special_weapon, weapon_range, weapon_power, weapon_rounds_per, weapon_iine
        a_players = []
        for player_x, player_layer in zip(x['a_players'], self.a_players):
            a_players.append(player_layer(
                player_x['weapon'], player_x['rank'], player_x['level'], player_x['sub_weapon'], player_x['special_weapon'],
                player_x['weapon_range'], player_x['weapon_power'], player_x['weapon_rounds_per'], player_x['weapon_iine']
            ))
        
        b_players = []
        for player_x, player_layer in zip(x['b_players'], self.b_players):
            b_players.append(player_layer(
                player_x['weapon'], player_x['rank'], player_x['level'], player_x['sub_weapon'], player_x['special_weapon'],
                player_x['weapon_range'], player_x['weapon_power'], player_x['weapon_rounds_per'], player_x['weapon_iine']
            ))
        
        a_team = torch.mean(torch.stack(a_players), axis=0)
        b_team = torch.mean(torch.stack(b_players), axis=0)
 
        data = {
            'lobby_mode': lobby_mode,
            'mode': mode,
            'stage': stage,
            'a_team': a_team,
            'b_team': b_team,
        }
        
        for block in self.transformer_blocks:
            data, attention_weight = block(data)
            self.attention_weights.append(attention_weight)
        
        output = torch.cat([data['lobby_mode'], data['mode'], data['stage'], data['a_team'], data['b_team']], axis=-1)
        output = self.out(output)
        
        if self.predict:
            output = self.out_softmax(output)       
        
        return output
    
