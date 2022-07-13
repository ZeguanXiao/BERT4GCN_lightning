import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        """
        :param text: size of (batch, paded_seq_len, embed_dim)
        :param adj: size of (batch, paded_seq_len, paded_seq_len)
        :return:
        """
        # (batch,seq_len,dim)*(dim,dim)->(batch,seq_len,dim)
        hidden = torch.matmul(text, self.weight)

        # (batch,seq_len,seq_len)->(batch,seq_len,1)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1

        # (batch,seq_len,seq_len)*(batch,seq_len,dim)->(batch,seq_len,dim)/(batch,seq_len,1)->
        output = torch.matmul(adj, hidden)

        output = output / denom

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCNWithPosition(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GCNWithPosition, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj, position_emb=None):
        """
        :param text: size of (batch, paded_seq_len, embed_dim)
        :param adj: size of (batch, paded_seq_len, paded_seq_len)
        :return:
        """
        # (batch,seq_len,seq_len)->(batch,seq_len,1)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1

        if position_emb is not None:
            hidden = torch.matmul(text, self.weight)

            # (batch,seq_len,seq_len,dim)*(dim,dim)->(batch,seq_len,seq_len,dim)
            pos_trans = torch.matmul(position_emb, self.weight)

            # (batch,1,seq_len,dim)+(batch,seq_len,seq_len,dim)->(batch,seq_len,seq_len,dim)
            hidden = hidden.unsqueeze(1) + pos_trans

            # (batch,seq_len,1,seq_len)*(batch,seq_len,seq_len,dim)->(batch,seq_len,1,dim)->(batch,seq_len,dim)
            output = torch.matmul(adj.unsqueeze(2), hidden).squeeze(2)
        else:
            hidden = torch.matmul(text, self.weight)  # (batch,seq_len,dim)*(dim,dim)->(batch,seq_len,dim)
            output = torch.matmul(adj, hidden)

        output = output / denom

        if self.bias is not None:
            return output + self.bias
        else:
            return output
