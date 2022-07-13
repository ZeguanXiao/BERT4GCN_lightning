# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.dynamic_rnn import DynamicLSTM


class BiLSTMEmbedding(nn.Module):
    """
    Glove + BiLSTM
    """
    def __init__(self, embedding_matrix, embed_dim, hidden_dim, freeze=True):
        super(BiLSTMEmbedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=freeze)
        self.lstm = DynamicLSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, text_raw_indices):
        batch_size, max_len = text_raw_indices.shape[0], text_raw_indices.shape[1]

        # (batch_size, max_len, 300)
        input_embed = self.embedding(text_raw_indices)

        input_lens = torch.sum(text_raw_indices != 0, dim=-1).cpu()
        # (batch_size, max_len of samples , hidden_dim*2)
        out, (_, _) = self.lstm(input_embed, input_lens)

        # pad to max_len
        feature = F.pad(out, [0, 0, 0, max_len - out.shape[1]], "constant", 0)

        return feature





