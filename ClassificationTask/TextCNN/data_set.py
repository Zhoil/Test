import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)  # 获取日志对象并赋值给logger


class TextCNNDataSet(Dataset):
    def __init__(self, vocab_file, max_len, label_file, path_file):  # 词汇表加载
        with open(vocab_file, "r", encoding="utf-8") as fh:  # 加载词汇表文件
            self.vocab = json.load(fh)
        with open(label_file, "r", encoding="utf-8") as fh:  # 加载标签文件
            self.label_dict = json.load(fh)
        self.label_number = len(self.label_dict)  # 计算长度
        self.max_len = max_len  # 最大长度
        self.vocab_size = len(self.vocab)  # 计算大小
        logger.info("进行数据预处理操作")
        self.data_set = self.convert_data_to_ids(path_file)  # 将文本数据转换为id显示

    def convert_data_to_ids(self, path_file):  # 转换文本为id
        self.data_set = []  # 储存转换后的数据
        with open(path_file, "r", encoding="utf-8") as data:
            for _, line in enumerate(tqdm(data, desc="iter", disable=False)):
                line = line.rstrip().split("\t")
                sample = {"text": "".join(line[:-1]), "label": self.label_dict[line[-1]]}
                sample = self.convert_sample_to_id(sample)
                self.data_set.append(sample)
        return self.data_set

    def convert_sample_to_id(self, sample):  # 将每个单词转换为对应的id，并存储在sample的input_ids中
        text = sample["text"].lower()  # 文本转换为小写
        tokens = text.split()  # 按空格分割文本为单词
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        input_ids = []  # 如上第一行，储存
        for token in tokens:
            if token in self.vocab:
                input_ids.append(self.vocab[token])
            else:
                input_ids.append(self.vocab["<UNK>"])
        input_ids = input_ids + [0] * (self.max_len - len(input_ids))
        sample["input_ids"] = input_ids  # 将变量 input_ids 的值赋给了字典 sample 中的 "input_ids" 键
        assert len(sample["input_ids"]) == self.max_len  # 检查 sample 字典中的 "input_ids" 键对应的值的长度与 self.max_len 是否相等
        return sample

    def __len__(self):  # 获取数据集的长度
        return len(self.data_set)

    def __getitem__(self, idx):  # 通过索引 idx 获取数据集中的一个样本
        instance = self.data_set[idx]
        return instance


def collate_func(batch_data):
    batch_size = len(batch_data)  # 获取这批数据的大小
    if batch_size == 0:
        return {}  # 大小为零，则返回一个空字典
    input_ids_list, labels_list = [], []  # 储存数据
    text_list = []  # 储存文本
    for instance in batch_data:
        input_ids_list.append(instance["input_ids"])
        labels_list.append(instance["label"])
        text_list.append(instance["text"])
    return {"input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "text": text_list}
