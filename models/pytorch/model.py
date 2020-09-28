import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dateutil.parser import parse as date_parse
from .datasets import XData

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        embedded = self.embed(x)
        out = self.norm(embedded)
        return out
    
class PlayerEmbedding(nn.Module):
    def __init__(self, weapon_embed: nn.Embedding, rank_embed: nn.Embedding):
        super().__init__()
        self.weapon_embed = weapon_embed
        self.rank_embed = rank_embed
        
    def forward(self, weapon, rank, level):
        weapon_embedded = self.weapon_embed(weapon)
        rank_embedded = self.rank_embed(rank)
        
        feat = torch.cat([weapon_embedded, rank_embedded, level.unsqueeze(-1)], axis=1)
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
        
        self.player_a1_cell = TransformerCell(player_dim)
        self.player_a2_cell = TransformerCell(player_dim)
        self.player_a3_cell = TransformerCell(player_dim)
        self.player_a4_cell = TransformerCell(player_dim)
        
        self.player_b1_cell = TransformerCell(player_dim)
        self.player_b2_cell = TransformerCell(player_dim)
        self.player_b3_cell = TransformerCell(player_dim)
        self.player_b4_cell = TransformerCell(player_dim)
 
    def forward(self, xdata: XData):
        lobby_mode, lobby_mode_attn = self.lobby_mode_cell(xdata.lobby_mode)
        mode, mode_attn = self.mode_cell(xdata.mode)
        stage, stage_attn = self.stage_cell(xdata.stage)
        
        player_a1, player_a1_attn = self.player_a1_cell(xdata.player_a1)
        player_a2, player_a2_attn = self.player_a2_cell(xdata.player_a2)
        player_a3, player_a3_attn = self.player_a3_cell(xdata.player_a3)
        player_a4, player_a4_attn = self.player_a4_cell(xdata.player_a4)
        
        player_b1, player_b1_attn = self.player_b1_cell(xdata.player_b1)
        player_b2, player_b2_attn = self.player_b2_cell(xdata.player_b2)
        player_b3, player_b3_attn = self.player_b3_cell(xdata.player_b3)
        player_b4, player_b4_attn = self.player_b4_cell(xdata.player_b4)
        
        output = XData(lobby_mode, mode, stage,
                       player_a1, player_a2, player_a3, player_a4,
                       player_b1, player_b2, player_b3, player_b4)
        attention_weights = {
            'lobby_mode': lobby_mode_attn,
            'mode': mode_attn,
            'stage': stage_attn,
            'player_a1': player_a1_attn,
            'player_a2': player_a2_attn,
            'player_a3': player_a3_attn,
            'player_a4': player_a4_attn,
            'player_b1': player_b1_attn,
            'player_b2': player_b2_attn,
            'player_b3': player_b3_attn,
            'player_b4': player_b4_attn,
        }
        
        return output, attention_weights
    
class SimpleTransformer(nn.Module):
    def __init__(self, n_lobby_modes: int, n_modes: int, n_stages: int, n_weapons: int, n_ranks: int):
        super().__init__()
        self.weapon_embed = Embedding(n_weapons, n_weapons)
        self.rank_embed = Embedding(n_ranks, n_ranks)
        self.lobby_mode_embed = Embedding(n_lobby_modes, n_lobby_modes)
        self.mode_embed = Embedding(n_modes, n_modes)
        self.stage_embed = Embedding(n_stages, n_stages)
        
        self.player_a1 = PlayerEmbedding(self.weapon_embed, self.rank_embed)
        self.player_a2 = PlayerEmbedding(self.weapon_embed, self.rank_embed)
        self.player_a3 = PlayerEmbedding(self.weapon_embed, self.rank_embed)
        self.player_a4 = PlayerEmbedding(self.weapon_embed, self.rank_embed)
        
        self.player_b1 = PlayerEmbedding(self.weapon_embed, self.rank_embed)
        self.player_b2 = PlayerEmbedding(self.weapon_embed, self.rank_embed)
        self.player_b3 = PlayerEmbedding(self.weapon_embed, self.rank_embed)
        self.player_b4 = PlayerEmbedding(self.weapon_embed, self.rank_embed)
        
        self.player_dim = self.weapon_embed.embedding_dim + self.rank_embed.embedding_dim + 1
        self.lobby_mode_dim = self.lobby_mode_embed.embedding_dim
        self.mode_dim = self.mode_embed.embedding_dim
        self.stage_dim = self.stage_embed.embedding_dim
        
        self.tf_block_1 = TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim)
        self.tf_block_2 = TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim)
        self.tf_block_3 = TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim)
        self.tf_block_4 = TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim)
        self.tf_block_5 = TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim)
        self.tf_block_6 = TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim)
        self.tf_block_7 = TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim)
        self.tf_block_8 = TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim)
        self.tf_block_9 = TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim)
        self.tf_block_10 = TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim)
        self.tf_block_11 = TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim)
        self.tf_block_12 = TransformerBlock(self.player_dim, self.lobby_mode_dim, self.mode_dim, self.stage_dim)
        
        self.transformer_blocks = [
            self.tf_block_1,
            self.tf_block_2,
            self.tf_block_3,
            self.tf_block_4,
            self.tf_block_5,
            self.tf_block_6,
            self.tf_block_7,
            self.tf_block_8,
            self.tf_block_9,
            self.tf_block_10,
            self.tf_block_11,
            self.tf_block_12,
        ]
        self.out = nn.Linear(self.lobby_mode_dim + self.mode_dim + self.stage_dim + self.player_dim * 8, 2)
        
        self.attention_weights = []
        
    def forward(self, x):
        self.attention_weights = []

        lobby_mode = self.lobby_mode_embed(x.lobby_mode)
        mode = self.mode_embed(x.mode)
        stage = self.stage_embed(x.stage)
        
        player_a1 = self.player_a1(x.a1_weapon, x.a1_rank, x.a1_level)
        player_a2 = self.player_a2(x.a2_weapon, x.a2_rank, x.a2_level)
        player_a3 = self.player_a3(x.a3_weapon, x.a3_rank, x.a3_level)
        player_a4 = self.player_a4(x.a4_weapon, x.a4_rank, x.a4_level)
        
        player_b1 = self.player_b1(x.b1_weapon, x.b1_rank, x.b1_level)
        player_b2 = self.player_b2(x.b2_weapon, x.b2_rank, x.b2_level)
        player_b3 = self.player_b3(x.b3_weapon, x.b3_rank, x.b3_level)
        player_b4 = self.player_b4(x.b4_weapon, x.b4_rank, x.b4_level)
        
        xdata = XData(lobby_mode, mode, stage,
                      player_a1, player_a2, player_a3, player_a4,
                      player_b1, player_b2, player_b3, player_b4)
        
        for block in self.transformer_blocks:
            xdata, attention_weight = block(xdata)
            self.attention_weights.append(attention_weight)
        
        output = torch.cat([xdata.lobby_mode, xdata.mode, xdata.stage, 
                            xdata.player_a1, xdata.player_a2, xdata.player_a3, xdata.player_a4,
                            xdata.player_b1, xdata.player_b2, xdata.player_b3, xdata.player_b4], axis=-1)
        output = self.out(output)
        
        return output
    
