import spacy
import json
from tqdm import tqdm
import numpy as np
import os

nlp = spacy.blank("en")


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]  # 对句子进行处理，然后将每个单词提取出来作为一个列表返回


def build_vocab(train_file, dev_file, word_vocab_file, char_vocab_file):
    word_vocab = {'<PAD>': 0, '<UNK>': 1}  # 初始化它们的索引为0和1
    char_vocab = {'<PAD>': 0, '<UNK>': 1}  # 初始化它们的索引为0和1
    for path in [train_file, dev_file]:  # 以迭代的方式读取训练文件和开发文件，加载数据集中的句子和问题
        with open(path, "r", encoding="utf-8") as fh:
            input_data = json.load(fh)["data"]
        for i, entry in enumerate(tqdm(input_data, desc="iter", disable=False)):
            for paragraph in entry["paragraphs"]:
                context = paragraph["context"].lower()  # 转换为小写
                context_tokens = word_tokenize(context)  # 分词
                for token in context_tokens:  # 遍历分词后的单词
                    if token not in word_vocab:
                        word_vocab[token] = len(word_vocab)  # 分配一个新的索引
                    for char in token:
                        if char not in char_vocab:
                            char_vocab[char] = len(char_vocab)  # 遍历其中的每个字符，并将字符添加到字符词汇表中
                for qa in paragraph["qas"]:
                    question = qa["question"].lower()  # 转换为小写
                    question_tokens = word_tokenize(question)  # 分词
                    for token in question_tokens:  # 同上
                        if token not in word_vocab:
                            word_vocab[token] = len(word_vocab)
                        for char in token:
                            if char not in char_vocab:
                                char_vocab[char] = len(char_vocab)
    
    with open(word_vocab_file, "w", encoding="utf-8") as fh:
        json.dump(word_vocab, fh, ensure_ascii=False, indent=4)  # 使用json.dump函数将词汇表对象保存到文件中
    with open(char_vocab_file, "w", encoding="utf-8") as fh:
        json.dump(char_vocab, fh, ensure_ascii=False, indent=4)  # 使用json.dump函数将词汇表对象保存到文件中
    return word_vocab, char_vocab  # 返回


def load_emb(w2v_file, vocab, emb_file):  # 打开预训练词嵌入文件
    w2v_dict = {}  # 设置一个空字典以储存
    with open(w2v_file, "r", encoding="utf-8") as fh:
        for i, line in enumerate(tqdm(fh, desc="iter", disable=False)):  # 循环逐行读取预训练词嵌入文件中的内容
            array = line.rstrip().split(' ')  # 以‘ ’分词
            word, vector = array[0], array[1:]
            vec_size = len(array[1:])
            w2v_dict[word] = list(map(float, vector))  # 将单词作为键，将数值列表作为值，存储在w2v_dict字典中
    emb_dict = {}  # 代码创建一个空字典emb_dict，用于存储词汇表（vocab）中的单词的词嵌入
    for k, v in vocab.items():
        if k in w2v_dict:
            emb_dict[k] = w2v_dict[k]  # 存在对应的词嵌入，则将其添加到emb_dict中
        else:
            emb_dict[k] = [np.random.normal(scale=0.1) for _ in range(vec_size)]  # 否则，生成一个随机的词嵌入向量，用于表示该单词
    idx2emb_dict = {idx: emb_dict[word] for word, idx in vocab.items()}  # 将词嵌入词典（emb_dict）转换为索引-词嵌入的字典（idx2emb_dict）
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]  # 将索引-词嵌入字典转换为词嵌入矩阵（emb_mat）
    fin = open(emb_file, "w", encoding="utf-8")  # 保存到文件
    json.dump(emb_mat, fin)
    fin.close()


if __name__ == '__main__':  # 执行程序
    train_file_path = "../data/train-v1.1.json"
    dev_file_path = "../data/dev-v1.1.json"
    word_vocab_file_path = "temp_data/word_vocab.json"
    char_vocab_file_path = "temp_data/char_vocab.json"
    word_vocab, char_vocab = build_vocab(train_file_path, dev_file_path, word_vocab_file_path, char_vocab_file_path)
    word_vector_file = "../data/glove.6B.100d.txt"
    if os.path.exists(word_vector_file):
        emb_file_path = "temp_data/emb.json"
        load_emb(word_vector_file, word_vocab, emb_file_path)