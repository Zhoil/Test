import numpy as np
import json
from tqdm import tqdm
import os
import argparse

def build_vocab(train_file, dev_file, vocab_file, label_file):  # 根据训练集和验证集构建词汇表，并将词汇表和标签字典保存到文件中
    vocab = {'<PAD>': 0, '<UNK>': 1}  # 初始化两个特殊的token：<PAD>和<UNK>，分别对应索引0和1
    label_dict = {}  # 初始化词向量字典，用于存储标签和对应的索引
    for path in [train_file, dev_file]:  # 处理训练集和处理验证集
        with open(path, "r", encoding="utf-8") as fh:
            for _, line in enumerate(tqdm(fh, desc="iter", disable=False)):
                line = line.rstrip().split("\t")  # 假设训练集中每行是一个样本，以空格分隔单词和标签
                text = "".join(line[:-1]).lower()  # 小写

                if line[-1] not in label_dict:
                    label_dict[line[-1]] = len(label_dict)  # 将标签添加到label_dict并分配一个对应的索引

                for token in text.split():  # 将文本分割为单词，并对每个单词进行处理
                    if token not in vocab:
                        vocab[token] = len(vocab)

    with open(vocab_file, "w", encoding="utf-8") as fh:  # 保存词汇表到文件
        json.dump(vocab, fh, ensure_ascii=False, indent=4)
    with open(label_file, "w", encoding="utf-8") as fh:  # 保存标签字典到文件
        json.dump(label_dict, fh, ensure_ascii=False, indent=4)
    return vocab


def load_emb(w2v_file, vocab, emb_file):
    # 从预训练词向量文件中加载词向量，保存到文件
    # w2v_file：预训练词向量文件路径
    # vocab：词汇表
    # emb_file：词向量保存路径

    w2v_dict = {}  # 初始化词向量字典
    with open(w2v_file, "r", encoding="utf-8") as fh:  # 保存词向量到文件
        for i, line in enumerate(tqdm(fh, desc="iter", disable=False)):
            array = line.rstrip().split(' ')  # 假设预训练词向量文件的每行包含一个词和其向量表示，以空格分隔
            word, vector = array[0], array[1:]  # 假设词向量是一个列表
            vec_size = len(array[1:])
            w2v_dict[word] = list(map(float, vector))
    emb_dict = {}
    for k, v in vocab.items():
        if k in w2v_dict:   # 如果词在词汇表中存在，则保存其词向量
            emb_dict[k] = w2v_dict[k]
        else:
            emb_dict[k] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
    idx2emb_dict = {idx: emb_dict[word] for word, idx in vocab.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    fin = open(emb_file, "w", encoding="utf-8")  # 保存词向量到文件
    json.dump(emb_mat, fin)
    fin.close()


def set_args():  # 设置命令行参数，控制代码的执行
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='SNIPS', type=str, help='')  # 指定任务类型
    parser.add_argument('--vocab_file', default='temp_data/vocab.json', type=str, help='')  # 用于指定词汇表文件路径
    parser.add_argument('--emb_file', default='temp_data/emb.json', type=str, help='')  # 用于指定嵌入向量文件路径
    parser.add_argument('--train_file', default='../data/{}/train.tsv', type=str, help='')  # 用于指定训练数据文件路径，{}会被具体的任务类型替换
    parser.add_argument('--dev_file', default='../data/{}/valid.tsv', type=str, help='')  # 用于指定验证数据文件路径，{}会被具体的任务类型替换
    parser.add_argument('--label_file', default='temp_data/{}/label.json', type=str, help='')  # 用于指定标签文件路径，{}会被具体的任务类型替换
    parser.add_argument('--word_vector_file', default='../data/glove.6B.100d.txt', type=str, help='')  # 用于指定词向量文件路径
    return parser.parse_args()


if __name__ == '__main__':  # 执行
    args = set_args()  # 调用控制运行的代码，获得命令行参数并赋给args变量
    args.train_file = args.train_file.format(args.task)  # 将{}替换为具体的任务类型
    args.dev_file = args.dev_file.format(args.task)  # 将{}替换为具体的任务类型
    args.label_file = args.label_file.format(args.task)  # 将{}替换为具体的任务类型
    vocab = build_vocab(args.train_file, args.dev_file, args.vocab_file, args.label_file)  # 构建词汇表，并返回构建完成的词汇表vocab
    if os.path.exists(args.word_vector_file):  # 判断args.word_vector_file文件是否存在
        load_emb(args.word_vector_file, vocab, args.emb_file)  # 加载词向量，并保存到args.emb_file中
