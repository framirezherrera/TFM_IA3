# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: TFM_UT.py
# ----------------------------------------------------------------------------------------------------------------------------------
# Descripción: Recurrent Transformer (RT) and Evolved Transformer (ET)
# Basado en: https://github.com/moon23k/Transformer_Variants/blob/main/model/components.py
#            https://github.com/moon23k/Transformer_Variants/blob/main/model/evolved.py
# ----------------------------------------------------------------------------------------------------------------------------------
# Implementado por: Felipe Ramírez Herrera
# Trabajo Final de Máster (TFM)
# Master de Inteligencia Artificial Avanzada y Aplicada (IA3)
# Universidad de Valencia / ADEIT
# Última revisión: 30/07/2024 
# ----------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import torch, math
import torch.nn as nn
import copy, math, torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F


class XT_Config:
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim = 256, hidden_dim = 256, pff_dim = 512, n_layers = 4, n_heads = 8,  dropout_ratio =  0.1,  max_len = 512, pad_id = 0, device = None): 
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.pff_dim = pff_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_ratio = dropout_ratio
        self.max_len = max_len
        self.pad_id = pad_id
        self.device = device

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        
        max_len = config.max_len # if config.task != 'summarization' else config.max_len * 4
        pe = torch.zeros(max_len, config.emb_dim)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.emb_dim, 2) * -(math.log(10000.0) / config.emb_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
        

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ModelBase(nn.Module):
    def __init__(self, config):
        super(ModelBase, self).__init__()

        #Attr Setup
        self.pad_id = config.pad_id
        self.device = config.device
        self.src_vocab_size = config.src_vocab_size
        self.tgt_vocab_size = config.tgt_vocab_size
        
        #Module Setup
        self.generator = nn.Linear(config.hidden_dim, config.tgt_vocab_size)


    def pad_mask(self, x):
        return x == self.pad_id


    def causal_mask(self, x):
        sz = x.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)        



class Embeddings(nn.Module):
    def __init__(self, vocab_size, config):
        super(Embeddings, self).__init__()

        self.tok_emb = nn.Embedding(vocab_size, config.emb_dim)
        self.scale = math.sqrt(config.emb_dim)
        self.pos_emb = PositionalEncoding(config)
        self.dropout = nn.Dropout(config.dropout_ratio)

        self.use_fc_layer = (config.emb_dim != config.hidden_dim)
        if self.use_fc_layer:
            self.fc = nn.Linear(config.emb_dim, config.hidden_dim)

    def forward(self, x):
        out = self.tok_emb(x) * self.scale
        out = self.pos_emb(out)

        if self.use_fc_layer:
            return self.dropout(self.fc(out))
        return self.dropout(out)


def generate_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = ( math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float32) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal =  signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)





class RecurrentEncoder(nn.Module):
    def __init__(self, vocab_size, config):
        super(RecurrentEncoder, self).__init__()    

        self.n_layers = config.n_layers
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.embeddings = Embeddings(vocab_size, config)
        
        max_len = config.max_len # if config.task != 'summarization' else config.max_len * 4
        
        self.time_signal = generate_signal(
            max_len, config.hidden_dim
        ).to(config.device)

        self.pos_signal = generate_signal(
            config.n_layers, config.hidden_dim
        ).to(config.device)
        

        self.layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            batch_first=True,
            norm_first=True
        )
        

    def forward(self, x, e_mask):
        x = self.embeddings(x)
        seq_len = x.size(1)

        for l in range(self.n_layers):
            x += self.time_signal[:, :seq_len, :]
            x += self.pos_signal[:, l, :].unsqueeze(1).repeat(1, seq_len, 1)
            x = self.layer(x, src_key_padding_mask=e_mask)
        
        return self.norm(x)




class RecurrentDecoder(nn.Module):
    def __init__(self, vocab_size, config):
        super(RecurrentDecoder, self).__init__()

        self.n_layers = config.n_layers
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.embeddings = Embeddings(vocab_size, config)
        
        self.time_signal = generate_signal(
            512, config.hidden_dim
        ).to(config.device)
        
        self.pos_signal = generate_signal(
            config.n_layers, config.hidden_dim
        ).to(config.device)

        self.layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            batch_first=True,
            norm_first=True
        )



    def forward(self, x, m, e_mask, d_mask):

        x = self.embeddings(x)

        seq_len = x.size(1)

        for l in range(self.n_layers):
            x += self.time_signal[:, :seq_len, :]
            x += self.pos_signal[:, l, :].unsqueeze(1).repeat(1, seq_len, 1)
            x = self.layer(
                tgt=x, memory=m,
                memory_key_padding_mask=e_mask, 
                tgt_mask=d_mask
            )

        return self.norm(x)




class RecurrentTransformer(ModelBase):
    def __init__(self,  config):
        super(RecurrentTransformer, self).__init__(config) 
        self.encoder = RecurrentEncoder(config.src_vocab_size, config)
        self.decoder = RecurrentDecoder(config.tgt_vocab_size, config)


        
    def forward(self, x, y):
        e_mask, d_mask = self.pad_mask(x), self.causal_mask(y) 
        memory = self.encoder(x, e_mask)
        dec_out = self.decoder(y, memory, e_mask, d_mask)
        logit = self.generator(dec_out)
        return logit






# ----------------------------------------------------------------------------------------------------------------------------------
# END OF FILE
# ----------------------------------------------------------------------------------------------------------------------------------