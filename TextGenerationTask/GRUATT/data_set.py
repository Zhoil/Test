import torch
import json
from tqdm import tqdm
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class GRUATTDataSet(Dataset):
    def __init__(self, src_lang, tgt_lang, vocab_file, src_max_len, tgt_max_len, path_file):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        with open("{}.{}.json".format(vocab_file, src_lang), "r", encoding="utf-8") as sfh:
            self.src_vocab = json.load(sfh)
        with open("{}.{}.json".format(vocab_file, tgt_lang), "r", encoding="utf-8") as tfh:
            self.tgt_vocab = json.load(tfh)
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)
        logger.info("进行数据预处理操作")
        self.src_data_set, self.tgt_data_set, self.src_text_data_set, self.tgt_text_data_set = self.convert_data_to_ids(path_file)

    def convert_data_to_ids(self, path_file):
        self.src_data_set = []
        self.tgt_data_set = []
        self.src_text_data_set = []
        self.tgt_text_data_set = []
        with open("{}.{}".format(path_file, self.src_lang), "r", encoding="utf-8") as src_data, \
            open("{}.{}".format(path_file, self.tgt_lang), "r", encoding="utf-8") as tgt_data:
            src_lines = src_data.readlines()
            tgt_lines = tgt_data.readlines()
            assert len(src_lines)== len(tgt_lines)
            for _, (src_line, tgt_line) in enumerate(zip(tqdm(src_lines, desc="iter", disable=False), tgt_lines)):
                src_sample = src_line.strip()
                tgt_sample = tgt_line.strip()
                src_input_ids, shifted_tgt_ids = self.convert_sample_to_id(src_sample, tgt_sample)
                if src_input_ids!=None and shifted_tgt_ids!=None:
                    self.src_data_set.append(src_input_ids)
                    self.tgt_data_set.append(shifted_tgt_ids)
                    self.src_text_data_set.append(src_sample)
                    self.tgt_text_data_set.append(tgt_sample)
        return self.src_data_set, self.tgt_data_set, self.src_text_data_set, self.tgt_text_data_set

    def convert_sample_to_id(self, src_sample, tgt_sample):
        src_tokens = src_sample.split()
        tgt_tokens = tgt_sample.split()
        if len(src_tokens) > self.src_max_len-1 or len(tgt_tokens) > self.tgt_max_len-2:
            return None, None
        src_input_ids = []
        for token in src_tokens:
            if token in self.src_vocab:
                src_input_ids.append(self.src_vocab[token])
            else:
                src_input_ids.append(self.src_vocab["<UNK>"])
        src_input_ids = src_input_ids + [self.src_vocab["<eos>"]]
        src_input_ids = src_input_ids + [self.src_vocab["<PAD>"]] * (self.src_max_len - len(src_input_ids))
        
        tgt_ids = []
        for token in tgt_tokens:
            if token in self.tgt_vocab:
                tgt_ids.append(self.tgt_vocab[token])
            else:
                tgt_ids.append(self.tgt_vocab["<UNK>"])
        shifted_tgt_ids = [self.tgt_vocab["<bos>"]] + tgt_ids + [self.tgt_vocab["<eos>"]]
        shifted_tgt_ids = shifted_tgt_ids + [self.tgt_vocab["<PAD>"]] * (self.tgt_max_len - len(shifted_tgt_ids))

        assert len(src_input_ids) == self.src_max_len
        assert len(shifted_tgt_ids) == self.tgt_max_len
        return src_input_ids, shifted_tgt_ids

    def __len__(self):
        return len(self.src_data_set)

    def __getitem__(self, idx):
        instance = {
            "src_input_ids": self.src_data_set[idx],
            "shifted_tgt_ids": self.tgt_data_set[idx],
            "src_text": self.src_text_data_set[idx],
            "tgt_text": self.tgt_text_data_set[idx],
        }
        return instance


def collate_func(batch_data):
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    src_input_ids_list, shifted_tgt_ids_list = [], []
    bitext_list = []
    for instance in batch_data:
        src_input_ids_list.append(instance["src_input_ids"])
        shifted_tgt_ids_list.append(instance["shifted_tgt_ids"])
        bitext_list.append({"src_text": instance["src_text"], "tgt_text": instance["tgt_text"]})
    return {"src_input_ids": torch.tensor(src_input_ids_list, dtype=torch.long),
            "shifted_tgt_ids": torch.tensor(shifted_tgt_ids_list, dtype=torch.long),
            "bitext": bitext_list}
