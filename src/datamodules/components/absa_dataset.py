# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.utils.data_utils import track_tokens


class ABSADataset(Dataset):
    def __init__(self, path, max_len, bert_tokenizer, text_tokenizer):
        with open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            lines = f.readlines()
        with open(path + '.graph', 'rb') as f:
            idx2gragh = pickle.load(f)

        all_data = []
        print('Building dataset...')
        for i in tqdm(range(0, len(lines), 3)):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            text = ' '.join([text_left, aspect, text_right])

            encoded_inputs = bert_tokenizer(text, aspect,
                                            padding='max_length',
                                            truncation=True,
                                            max_length=max_len,
                                            return_token_type_ids=True,
                                            return_tensors="pt")
            input_ids = encoded_inputs['input_ids'][0]
            token_type_ids = encoded_inputs['token_type_ids'][0]
            attention_mask = encoded_inputs['attention_mask'][0]

            token_start, token_start_mask = track_tokens(text.split(), max_len, bert_tokenizer)

            # pad graph adj matrix
            dependency_graph = idx2gragh[i]
            dependency_graph = np.pad(dependency_graph, ((0, max_len - dependency_graph.shape[0]), (0, max_len - dependency_graph.shape[0])), mode='constant')

            # polarity
            polarity = lines[i + 2].strip()
            polarity = int(polarity) + 1

            text_raw_indices = text_tokenizer.text_to_sequence(text)
            text_left_indices = text_tokenizer.text_to_sequence(text_left)
            aspect_indices = text_tokenizer.text_to_sequence(aspect)
            left_context_len = np.sum(text_left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_in_text = [left_context_len.item(), (left_context_len + aspect_len - 1).item()]
            aspect_in_text_mask = torch.zeros(max_len, dtype=torch.long)
            aspect_in_text_mask[left_context_len.item(): (left_context_len + aspect_len).item()] = 1

            assert torch.sum(token_start_mask) == sum((text_raw_indices != 0))
            data = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'dependency_graph': torch.tensor(dependency_graph),
                    'polarity': torch.tensor(polarity),
                    'token_starts': token_start,
                    'token_start_mask': token_start_mask,
                    'text_raw_indices': torch.tensor(text_raw_indices, dtype=torch.long),
                    'aspect_in_text': torch.tensor(aspect_in_text, dtype=torch.long),
                    'aspect_in_text_mask': aspect_in_text_mask,
                }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
