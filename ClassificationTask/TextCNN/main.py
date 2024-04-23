import os
from model import TextCNN
from data_set import TextCNNDataSet
from train import train
import argparse
import random
import torch
import numpy as np


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--task', default='SNIPS', type=str, help='')
    parser.add_argument('--vocab_file', default='temp_data/vocab.json', type=str, help='')
    parser.add_argument('--emb_file', default='temp_data/emb.json', type=str, help='')
    parser.add_argument('--train_file', default='../data/{}/train.tsv', type=str, help='')
    parser.add_argument('--dev_file', default='../data/{}/valid.tsv', type=str, help='')
    parser.add_argument('--label_file', default='temp_data/{}/label.json', type=str, help='')
    parser.add_argument('--vocab_size', default=10, type=int, help='')  # 指定词汇表大小
    parser.add_argument('--vec_size', default=100, type=int, help='')  # 指定词向量维度
    parser.add_argument('--filter_num', default=100, type=int, help='')  # 指定卷积层中的过滤器数量
    parser.add_argument('--in_channels', default=1, type=int, help='')  # 指定输入通道数
    parser.add_argument('--kernels', default=[3, 4, 5], type=list, help='')  # 指定卷积核的大小
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='')  # 指定dropout的概率
    parser.add_argument('--num_train_epochs', default=5, type=int, help='')  # 指定训练的轮数
    parser.add_argument('--train_batch_size', default=50, type=int, help='')  # 指定训练批次的大小
    parser.add_argument('--dev_batch_size', default=16, type=int, help='')  # 指定验证批次的大小
    parser.add_argument('--label_number', default=2, type=int, help='')  # 指定标签的数量
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='')  # 指定学习率
    parser.add_argument('--output_dir', default='output_dir/{}/', type=str, help='')  # 指定输出目录路径,{}会被具体的任务类型替换
    parser.add_argument('--seed', type=int, default=42, help='')  # 指定随机种子
    parser.add_argument('--max_len', type=int, default=128, help='')  # 指定文本的最大长度
    return parser.parse_args()  # 解析命令行参数，并返回解析结果


def main():
    args = set_args()  # data_helper 判断torch.cuda.is_available()是否为真
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")  # 根据系统环境和命令行参数选择合适的设备进行计算，以便在有可用的GPU时利用GPU加速模型训练和推理

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    args.train_file = args.train_file.format(args.task)
    args.dev_file = args.dev_file.format(args.task)     
    args.output_dir = args.output_dir.format(args.task) 
    args.label_file = args.label_file.format(args.task)
    
    train_data = TextCNNDataSet(args.vocab_file, args.max_len, args.label_file, args.train_file)
    dev_data = TextCNNDataSet(args.vocab_file, args.max_len, args.label_file, args.dev_file)
    args.vocab_size = train_data.vocab_size
    args.label_number = train_data.label_number

    model = TextCNN(args, args.emb_file)

    train(args, model, device, train_data, dev_data)

if __name__ == '__main__':
    main()
