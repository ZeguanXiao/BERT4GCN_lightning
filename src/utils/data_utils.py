# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import torch

from src.utils import get_project_root


def track_tokens(tokens, max_len, tokenizer):
    """Segment each token into subwords while keeping track of
    token boundaries.
    Parameters
    ----------
    tokens: A list of strings, representing input tokens.
    Returns
    ----------
    A tuple consisting of:
        - token_start_mask:
        An array with size (max_len) in which word starts tokens is 1 and all other subwords is 0.
        - token_start:
        An array of indices into the list of subwords, indicating
        that the corresponding subword is the start of a new
        token. For example, [1, 3, 4, 7] means that the subwords
        1, 3, 4, 7 are token starts, while all other subwords
        (0, 2, 5, 6, 8...) are in or at the end of tokens.
        This list allows selecting Bert hidden states that
        represent tokens, which is necessary in sequence
        labeling.
    """
    subwords = list(map(tokenizer.tokenize, tokens))
    subword_lengths = list(map(len, subwords))
    token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])

    token_start = torch.zeros(max_len, dtype=torch.long)
    token_start[0:len(token_start_idxs)] = torch.tensor(token_start_idxs)

    token_start_mask = torch.zeros(max_len, dtype=torch.long)
    token_start_mask[token_start_idxs] = 1

    return token_start, token_start_mask


def pad_and_truncate(sequence, max_len, dtype='int32', padding='post', truncating='post', value=0):
    x = (np.ones(max_len) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-max_len:]
    else:
        trunc = sequence[:max_len]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_len, lower=True):
        self.lower = lower
        self.max_len = max_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_len, padding=padding, truncating=truncating)


def build_tokenizer(fnames, max_len, dat_fname):
    """
    :param fnames: directory to data file
    :param max_len:
    :param dat_fname: filename to an exist tokenizer or the path to save a newly build tokenizer
    :return: tokenizer
    """
    # try load exist tokenizer
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))  # pickle the tokenizer to a file 'dat_fname'
    return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = os.path.join(get_project_root(), 'glove/glove.840B.300d.txt')
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        embedding_matrix[0, :] = np.zeros((1, embed_dim))
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


