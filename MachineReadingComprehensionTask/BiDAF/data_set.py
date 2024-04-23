import torch
import json
from tqdm import tqdm
from torch.utils.data import Dataset
import logging
from data_helper import word_tokenize
import numpy as np

logger = logging.getLogger(__name__)  # 创建一个日志记录器，以便在代码中进行日志输出


class BiDAFDataSet(Dataset):

    def __init__(self, word_vocab_file, char_vocab_file, context_max_len, query_max_len, word_max_len, path_file):  # 初始化类的实例对象的属性
        with open(word_vocab_file, "r", encoding="utf-8") as fh:
            self.word_vocab = json.load(fh)
        with open(char_vocab_file, "r", encoding="utf-8") as fh:
            self.char_vocab = json.load(fh)
        # 打开word_vocab_file和char_vocab_file文件，并通过json.load(fh)加载文件中的词汇表数据
        self.context_max_len = context_max_len  # 指定上下文
        self.query_max_len = query_max_len  # 查询
        self.word_max_len = word_max_len  # 词的最大长度
        self.word_vocab_size = len(self.word_vocab)  # 词汇表的长度
        self.char_vocab_size = len(self.char_vocab)  # 词汇表的长度
        logger.info("进行数据预处理操作")  # 记录日志信息
        self.data_set = self.convert_data_to_ids(path_file)  # 将文件路径path_file作为参数传递给方法，并将结果存储在self.data_set属性中

    def convert_data_to_ids(self, path_file):  # 将原始数据转换为ID格式的数据
        self.data_set = []  # 创建一个空列表self.data_set，用于存储转换后的数据
        with open(path_file, "r", encoding="utf-8") as fh:  # 使用json.load(fh)加载path_file文件中的数据
            input_data = json.load(fh)["data"]  # 将数据存储在input_data变量中
        for i, entry in enumerate(tqdm(input_data, desc="iter", disable=False)):
            for paragraph in entry["paragraphs"]:  # 获取
                context = paragraph["context"].lower()  # 转换为小写
                context_tokens = word_tokenize(context)  # 将上下文分词为词级别的列
                context_chars = [list(token) for token in context_tokens]  # 将每个词切分为字符级别的列表
                spans = self.convert_idx(context, context_tokens)
                for qa in paragraph["qas"]:
                    question = qa["question"].lower()  # 使用相同的方式处理question
                    question_tokens = word_tokenize(question)
                    question_chars = [list(token) for token in question_tokens]
                    answer = qa["answers"][0]  # 获取问题的答案
                    answer_text = answer["text"].lower()  # 获取问题的答案文本
                    answer_start = answer['answer_start']  # 答案的起始位置
                    answer_end = answer_start + len(answer_text)  # 答案结束位置
                    answer_span = []  # 进一步处理段落的段落标记（spans），以判断答案所在的分段范围
                    for idx, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(idx)
                    start_label, end_label = answer_span[0], answer_span[-1]
                    sample = {"context_tokens": context_tokens, "context_chars": context_chars,
                              "question_tokens": question_tokens, "question_chars": question_chars,
                              "start_label": start_label, "end_label": end_label, "qas_id": qa["id"],
                              "answer_texts": [answer["text"] for answer in qa["answers"]]}
                    # 构建一个样本（sample）字典，包含上下文的词级别和字符级别表示，问题的词级别和字符级别表示，以及答案的起始和结束标签，问题的id和答案文本
                    if end_label > self.context_max_len:  # 大于指定的上下文最大长度，则忽略该样本，并继续下一个样本
                        print(qa["id"])
                        continue
                    sample = self.convert_sample_to_id(sample)  # 将样本转换为id格式
                    self.data_set.append(sample)  # 将转换后的样本添加到self.data_set列表中
        return self.data_set  # 返回

    def convert_sample_to_id(self, sample):
        context_input_ids = np.zeros([self.context_max_len], dtype=np.int32)
        context_char_input_ids = np.zeros([self.context_max_len, self.word_max_len], dtype=np.int32)
        query_input_ids = np.zeros([self.query_max_len], dtype=np.int32)
        query_char_input_ids = np.zeros([self.query_max_len, self.word_max_len], dtype=np.int32)
        # 创建了四个名为context_input_ids、context_char_input_ids、query_input_ids和query_char_input_ids的numpy数组，用于存储转换后的id表示
        for i, token in enumerate(sample["context_tokens"]):  # 循环遍历样本的上下文词
            if i == self.context_max_len:
                break
            context_input_ids[i] = self._get_word(token)  # 根据词和字符获取相应的ID，并将ID赋值给对应的数组元素
        for i, token in enumerate(sample["question_tokens"]):  # 循环遍历样本的问题词
            if i == self.query_max_len:
                break
            query_input_ids[i] = self._get_word(token)  # 根据词和字符获取相应的ID，并将ID赋值给对应的数组元素
        for i, token in enumerate(sample["context_chars"]):  # 循环遍历样本的上下文字符
            if i == self.context_max_len:
                break
            for j, char in enumerate(token):
                if j == self.word_max_len:
                    break  # 字符的索引i超过了指定的上限，则跳出内层循环
                context_char_input_ids[i, j] = self._get_char(char)  # 根据词和字符获取相应的ID，并将ID赋值给对应的数组元素
        for i, token in enumerate(sample["question_chars"]):  # 循环遍历样本的问题字符
            if i == self.query_max_len:
                break
            for j, char in enumerate(token):
                if j == self.word_max_len:
                    break  # 词中字符的索引j超过了指定的上限，则跳出内层循环
                query_char_input_ids[i, j] = self._get_char(char)  # 根据词和字符获取相应的ID，并将ID赋值给对应的数组元素
        sample["context_input_ids"] = context_input_ids  # 赋值给样本字典的相应属
        sample["context_char_input_ids"] = context_char_input_ids  # 赋值给样本字典的相应属
        sample["query_input_ids"] = query_input_ids  # 赋值给样本字典的相应属
        sample["query_char_input_ids"] = query_char_input_ids  # 赋值给样本字典的相应属
        return sample  # 返回样本

    def _get_word(self, word):
        if word in self.word_vocab:
            return self.word_vocab[word]  # 如果在词汇表中找到了对应的词，代码返回该词的ID
        return 1  # 否则返回 1

    def _get_char(self, char):
        if char in self.char_vocab:
            return self.char_vocab[char]  # 如果在词汇表中找到了对应的词，代码返回该词的ID
        return 1  # 否则返回 1

    @staticmethod
    def convert_idx(text, tokens):
        current = 0  # 初始化变量为0，用于表示当前位置
        spans = []  # 创建一个空列表spans，用于存储每个词的起始和结束位置
        for token in tokens:  # 循环遍历
            current = text.find(token, current)  # 起始位置
            spans.append((current, current + len(token)))  # 将该起始位置与当前起始位置相对应的结束位置添加到spans列表中
            current += len(token)
        return spans  # 返回

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance  # 返回对应索引位置的样本实例


def collate_func(batch_data):
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    context_input_ids_list, context_char_input_ids_list = [], []
    query_input_ids_list, query_char_input_ids_list = [], []
    start_labels_list, end_labels_list = [], []
    sample_list = []
    for instance in batch_data:
        context_input_ids_list.append(instance["context_input_ids"])
        context_char_input_ids_list.append(instance["context_char_input_ids"])
        query_input_ids_list.append(instance["query_input_ids"])
        query_char_input_ids_list.append(instance["query_char_input_ids"])
        start_labels_list.append(instance["start_label"])
        end_labels_list.append(instance["end_label"])
        sample_list.append(instance)
    return {"context_input_ids": torch.tensor(context_input_ids_list, dtype=torch.long),
            "context_char_input_ids": torch.tensor(context_char_input_ids_list, dtype=torch.long),
            "query_input_ids": torch.tensor(query_input_ids_list, dtype=torch.long),
            "query_char_input_ids": torch.tensor(query_char_input_ids_list, dtype=torch.long),
            "start_labels": torch.tensor(start_labels_list, dtype=torch.long),
            "end_labels": torch.tensor(end_labels_list, dtype=torch.long),
            "sample": sample_list}
