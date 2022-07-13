# -*- coding: utf-8 -*-
import os
from typing import Any, List, Tuple
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.models.model_wrapper import ModelWrapper
from src.models.components.gcn import GCNWithPosition
from src.models.components.bilstm_embedding import BiLSTMEmbedding
from src.models.components.relativa_position import RelativePosition
from src.utils.data_utils import build_embedding_matrix


class BERT4GCN(ModelWrapper):
    def __init__(self,
                 dm_config: DictConfig,
                 word2idx: dict,
                 model_name_or_path: str = 'bert-base-uncased',
                 bert_layers: Tuple = (1, 5, 9, 12),
                 bert_dim: int = 768,
                 emb_dim: int = 300,
                 hidden_dim: int = 300,
                 upper: float = 0.25,
                 lower: float = 0.01,
                 window: int = 3,
                 gnn_drop: float = 0.8,
                 guidance_drop: float = 0.8,
                 freeze_emb: bool = True,
                 bert_learning_rate: float = 0.00002,
                 others_learning_rate: float = 0.001,
                 weight_decay: float = 0.01,
                 **kwargs):
        super(BERT4GCN, self).__init__()
        self.save_hyperparameters(logger=False)

        embedding_matrix = self._build_embedding_matrix()
        self.lstm_emb = BiLSTMEmbedding(embedding_matrix, emb_dim, hidden_dim, freeze=freeze_emb)
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Linear(hidden_dim * 2, 3)

        self.position_emb = RelativePosition(hidden_dim * 2, window)

        self.gnn = nn.ModuleList()
        self.guidance_trans = nn.ModuleList()
        for _ in range(len(bert_layers)):
            self.guidance_trans.append(nn.Linear(bert_dim, hidden_dim * 2))
            self.gnn.append(GCNWithPosition(hidden_dim * 2, hidden_dim * 2))

        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.gnn_drop = nn.Dropout(gnn_drop)
        self.guidance_drop = nn.Dropout(guidance_drop)

        self.reset_parameter()

    def _build_embedding_matrix(self):
        embedding_matrix_path = os.path.join(self.hparams.dm_config.cache_dir, f'{self.hparams.dm_config.dataset}_embedding_matrix.dat')
        embedding_matrix = build_embedding_matrix(self.hparams.word2idx,
                                                  embed_dim=self.hparams.emb_dim,
                                                  dat_fname=embedding_matrix_path)
        return embedding_matrix

    def forward(self, x):
        input_ids, attention_mask, token_type_ids, adj, token_starts, token_starts_mask, text_raw_indices, aspect_in_text, aspect_in_text_mask = x
        batch_size, max_len = text_raw_indices.shape[0], text_raw_indices.shape[1]

        # encode text
        feature = self.lstm_emb(text_raw_indices)
        outputs = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True, output_attentions=True)

        # select hidden_states of word start tokens
        stack_hidden_states = torch.stack(outputs.hidden_states)
        hidden_states_list = []
        for i in range(batch_size):
            start_tokens_hidden_states = torch.index_select(stack_hidden_states[:, i], dim=1, index=token_starts[i])
            hidden_states_list.append(start_tokens_hidden_states)
        guidance_states = torch.stack(hidden_states_list, dim=1)

        # select attention weights of word start tokens
        stack_attentions = torch.stack(outputs.attentions).clone().detach().mean(dim=2)
        attentions_list = []
        for i in range(batch_size):
            sample_attentions = stack_attentions[:, i]  # (n,max_len,max_len)
            sample_attentions = sample_attentions * token_starts_mask[i].view(max_len, 1) * token_starts_mask[i].view(1, max_len)
            start_tokens_attentions_row2col = torch.index_select(sample_attentions, dim=1, index=token_starts[i])
            start_tokens_attentions_col2row = torch.index_select(start_tokens_attentions_row2col, dim=2, index=token_starts[i])
            attentions_list.append(start_tokens_attentions_col2row)
        guidance_attentions = torch.stack(attentions_list, dim=1)

        pos = self.position_emb(max_len, max_len).unsqueeze(0).expand(batch_size, -1, -1, -1)

        for index, layer in enumerate(self.hparams.bert_layers):
            layer_hidden_states = guidance_states[layer]
            guidance = F.relu(self.guidance_trans[index](layer_hidden_states))
            node_embeddings = self.guidance_drop(guidance) + feature
            feature = self.layer_norm(node_embeddings)

            if index < len(self.hparams.bert_layers) - 1:
                layer_attentions = guidance_attentions[layer - 1]  # (batch, seq, seq)
                upper_att_adj = torch.gt(layer_attentions, self.hparams.upper)
                enhanced_adj = torch.logical_or(adj, upper_att_adj)
                lower_att_adj = torch.le(layer_attentions, self.hparams.lower)
                enhanced_adj = torch.logical_and(enhanced_adj, ~lower_att_adj)
                gnn_out = F.relu(self.gnn[index](feature, enhanced_adj.float(), pos))
                feature = self.gnn_drop(gnn_out)

        # (bs,seq,dim) * (bs,seq,1)
        aspects = (feature * aspect_in_text_mask.unsqueeze(2)).sum(dim=1) / aspect_in_text_mask.sum(dim=1, keepdim=True)
        logits = self.classifier(aspects)
        return logits

