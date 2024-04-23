import os
from model import BiDAFModel
from data_set import BiDAFDataSet
from train import train
import argparse
import random
import torch
import numpy as np


def set_args():  # 作用同第一个 Classifiction 类似
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--word_vocab_file', default='temp_data/word_vocab.json', type=str, help='')
    parser.add_argument('--char_vocab_file', default='temp_data/char_vocab.json', type=str, help='')
    parser.add_argument('--emb_file', default='temp_data/emb.json', type=str, help='')
    parser.add_argument('--train_file', default='../data/train-v1.1.json', type=str, help='')
    parser.add_argument('--dev_file', default='../data/dev-v1.1.json', type=str, help='')
    parser.add_argument('--word_vocab_size', default=10, type=int, help='')
    parser.add_argument('--char_vocab_size', default=10, type=int, help='')
    parser.add_argument('--word_vec_size', default=100, type=int, help='')
    parser.add_argument('--char_vec_size', default=25, type=int, help='')
    parser.add_argument('--hidden_size', default=100, type=int, help='')
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='')
    parser.add_argument('--train_batch_size', default=8, type=int, help='')
    parser.add_argument('--dev_batch_size', default=16, type=int, help='')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='')
    parser.add_argument('--output_dir', default='output_dir/', type=str, help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--context_max_len', type=int, default=512, help='')
    parser.add_argument('--query_max_len', type=int, default=64, help='')
    parser.add_argument('--word_max_len', type=int, default=8, help='')
    return parser.parse_args()


def main():
    args = set_args()  # 设置参数
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")  # 初始化设备

    if args.seed:  # 设置随机种子
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    train_data = BiDAFDataSet(args.word_vocab_file, args.char_vocab_file, args.context_max_len, args.query_max_len,
                              args.word_max_len, args.train_file)  # 加载训练
    dev_data = BiDAFDataSet(args.word_vocab_file, args.char_vocab_file, args.context_max_len, args.query_max_len,
                            args.word_max_len, args.dev_file)  # 加载验证数据
    args.word_vocab_size = train_data.word_vocab_size  # 设置词汇表大小
    args.char_vocab_size = train_data.char_vocab_size  # 设置词汇表大小

    model = BiDAFModel(args, args.emb_file)  # 构建模型

    train(args, model, device, train_data, dev_data)  # 进行训练


if __name__ == '__main__':
    main()
