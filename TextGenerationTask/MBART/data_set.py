import torch
import json
from tqdm import tqdm
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class MBARTDataSet(Dataset):

    def __init__(self, tokenizer, src_lang, tgt_lang, src_max_len, tgt_max_len, path_file):
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.bos_token_id = self.tokenizer.bos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.src_lang_code_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.src_lang)
        self.tgt_lang_code_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.src_data_set, self.tgt_data_set, self.src_text_data_set, self.tgt_text_data_set = self.convert_data_to_ids(path_file)

    def convert_data_to_ids(self, path_file):
        self.src_data_set = []
        self.tgt_data_set = []
        self.src_text_data_set = []
        self.tgt_text_data_set = []
        with open("{}.{}".format(path_file, self.src_lang), "r", encoding="utf-8") as src_data , \
            open("{}.{}".format(path_file, self.tgt_lang), "r", encoding="utf-8") as tgt_data:
            src_lines = src_data.readlines()
            tgt_lines = tgt_data.readlines()
            assert len(src_lines)==len(tgt_lines)
            for i, (src_line, tgt_line) in enumerate(zip(tqdm(src_lines, desc="iter", disable=False), tgt_lines)):
                src_sample = src_line.strip()
                tgt_sample = tgt_line.strip()
                src_input, tgt_label_ids = self.convert_sample_to_id(src_sample, tgt_sample)
                if src_input!=None and tgt_label_ids!=None:
                    self.src_data_set.append(src_input)
                    self.tgt_data_set.append(tgt_label_ids)
                    self.src_text_data_set.append(src_sample)
                    self.tgt_text_data_set.append(tgt_sample)
        return self.src_data_set, self.tgt_data_set, self.src_text_data_set, self.tgt_text_data_set

    def convert_sample_to_id(self, src_sample, tgt_sample):
        src_tokens = self.tokenizer.tokenize(src_sample)
        tgt_tokens = self.tokenizer.tokenize(tgt_sample)
        if len(src_tokens) > self.src_max_len-2 or len(tgt_tokens) > self.tgt_max_len-2:
            return None, None
        src_input_ids = []
        src_input_ids = [self.src_lang_code_id] + self.tokenizer.convert_tokens_to_ids(src_tokens) + [self.eos_token_id]
        src_attention_mask = [1] * len(src_input_ids) + [0] * (self.src_max_len - len(src_input_ids))
        src_input_ids = src_input_ids + [self.pad_token_id] * (self.src_max_len - len(src_input_ids))
        src_input = {"input_ids": src_input_ids, "attention_mask": src_attention_mask}

        tgt_label_ids = [self.tgt_lang_code_id] + self.tokenizer.convert_tokens_to_ids(tgt_tokens) + [self.eos_token_id]
        tgt_label_ids = tgt_label_ids + [self.pad_token_id] * (self.tgt_max_len - len(tgt_label_ids))
        return src_input, tgt_label_ids 

    def __len__(self):
        assert len(self.src_data_set)==len(self.tgt_data_set)
        return len(self.src_data_set)

    def __getitem__(self, idx):
        instance = {
            "src_input": self.src_data_set[idx],
            "tgt_label_ids": self.tgt_data_set[idx],
            "src_text": self.src_text_data_set[idx],
            "tgt_text": self.tgt_text_data_set[idx],
        }
        return instance


def collate_func(batch_data):
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    src_input_ids_list, attention_mask_list, tgt_label_ids_list = [], [], []
    bitext_list = []
    for instance in batch_data:
        src_input_ids_list.append(instance["src_input"]["input_ids"])
        attention_mask_list.append(instance["src_input"]["attention_mask"])
        tgt_label_ids_list.append(instance["tgt_label_ids"])
        bitext_list.append({"src_text": instance["src_text"], "tgt_text": instance["tgt_text"]})
    return {"src_input_ids": torch.tensor(src_input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "tgt_label_ids": torch.tensor(tgt_label_ids_list, dtype=torch.long),
            "bitext": bitext_list}
