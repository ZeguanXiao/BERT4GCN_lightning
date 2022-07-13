# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
from spacy.tokens import Doc
from tqdm import tqdm


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1
    return matrix


def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    filename = filename + '.graph'

    fout = open(filename, 'wb')
    for i in tqdm(range(0, len(lines), 3)):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        text = text_left + ' ' + aspect + ' ' + text_right
        adj_matrix = dependency_adj_matrix(text.strip())
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)        
    fout.close() 


nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

if __name__ == '__main__':
    process('data/acl-14-short-data/train.raw')
    process('data/acl-14-short-data/test.raw')
    process('data/semeval14/restaurant_train.raw')
    process('data/semeval14/restaurant_test.raw')
    process('data/semeval14/laptop_train.raw')
    process('data/semeval14/laptop_test.raw')
