import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)

class BERTDataSet(Dataset):
    def __init__(self, tokenizer, max_len, label_file, path_file):
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(label_file, "r", encoding="utf-8") as fh:
            self.label_dict = json.load(fh)
        self.label_number = len(self.label_dict)
        self.data_set = self.convert_data_to_ids(path_file)

    def convert_data_to_ids(self, path_file):
        self.data_set = []
        with open(path_file, "r", encoding="utf-8") as fh:
            for _, line in enumerate(tqdm(fh, desc="iter", disable=False)):
                line = line.rstrip().split("\t")
                sample = {"text": "".join(line[:-1]), "label": self.label_dict[line[-1]]}
                sample = self.convert_sample_to_id(sample)
                self.data_set.append(sample)
        return self.data_set
    
    def convert_sample_to_id(self, sample):
        text = sample["text"]
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_len - 2:
            tokens = tokens[:self.max_len - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
        attention_mask = [1] * len(input_ids)
        padding_id = self.tokenizer.convert_tokens_to_ids(['[PAD]'])
        input_ids = input_ids + padding_id * (self.max_len - len(input_ids))
        attention_mask = attention_mask + [0] * (self.max_len - len(attention_mask))
        token_type_ids = [0] * len(input_ids)
        position_ids = list(np.arange(len(input_ids)))
        sample["input_ids"] = input_ids
        sample["attention_mask"] = attention_mask
        sample["token_type_ids"] = token_type_ids
        sample["position_ids"] = position_ids
        assert len(sample["input_ids"]) == self.max_len
        assert len(sample["input_ids"]) == len(sample["attention_mask"])
        assert len(sample["input_ids"]) == len(sample["token_type_ids"])
        assert len(sample["input_ids"]) == len(sample["position_ids"])
        return sample

    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance

def collate_func(batch_data):
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    input_ids_list, attention_mask_list, token_type_ids_list, position_ids_list, labels_list = [], [], [], [], []
    text_list = []
    for instance in batch_data:
        input_ids_list.append(instance["input_ids"])
        attention_mask_list.append(instance["attention_mask"])
        token_type_ids_list.append(instance["token_type_ids"])
        position_ids_list.append(instance["position_ids"])
        labels_list.append(instance["label"])
        text_list.append(instance["text"])
    return {"input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids_list, dtype=torch.long),
            "position_ids": torch.tensor(position_ids_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "text": text_list}
