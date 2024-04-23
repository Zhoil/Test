import os
from model import BERTQAModel
from train import train
from data_set import BERTQADataSet
from transformers import BertTokenizer
import torch
import argparse
import random
import numpy as np


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--pre_train_model', default='../data/bert_base/', type=str, help='')
    parser.add_argument('--vocab_path', default='../data/bert_base/vocab.txt', type=str, help='')
    parser.add_argument('--train_file', default='../data/train-v1.1.json', type=str, help='')
    parser.add_argument('--dev_file', default='../data/dev-v1.1.json', type=str, help='')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='')
    parser.add_argument('--train_batch_size', default=4, type=int, help='')
    parser.add_argument('--dev_batch_size', default=8, type=int, help='')
    parser.add_argument('--hidden_size', default=768, type=int, help='')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='')
    parser.add_argument('--output_dir', default='output_dir/', type=str, help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--max_len', type=int, default=512, help='')
    parser.add_argument('--query_max_len', type=int, default=128, help='')
    return parser.parse_args()


def main():
    args = set_args()
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)


    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)

    train_data = BERTQADataSet(tokenizer, args.max_len, args.query_max_len, args.train_file)
    dev_data = BERTQADataSet(tokenizer, args.max_len, args.query_max_len, args.dev_file)

    model = BERTQAModel(args)
    train(args, model, device, train_data, dev_data)


if __name__ == '__main__':
    main()
