import os
from model import BERTClassificationModel
from train import train
from data_set import BERTDataSet, collate_func
from transformers import BertTokenizer
import torch
import argparse
import random
import numpy as np


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='')
    parser.add_argument('--task', default='SNIPS', type=str, help='')
    parser.add_argument('--pre_train_model', default='../data/bert_base/', type=str, help='')
    parser.add_argument('--vocab_path', default='../data/bert_base/vocab.txt', type=str, help='')
    parser.add_argument('--train_file', default='../data/{}/train.tsv', type=str, help='')
    parser.add_argument('--dev_file', default='../data/{}/valid.tsv', type=str, help='')
    parser.add_argument('--label_file', default='temp_data/{}/label.json', type=str, help='')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='')
    parser.add_argument('--train_batch_size', default=8, type=int, help='')
    parser.add_argument('--dev_batch_size', default=8, type=int, help='')
    parser.add_argument('--label_number', default=3, type=int, help='')
    parser.add_argument('--hidden_size', default=768, type=int, help='')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='')
    parser.add_argument('--output_dir', default='output_dir/{}/', type=str, help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--max_len', type=int, default=128, help='')
    return parser.parse_args()


def main():
    args = set_args()
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)

    args.train_file = args.train_file.format(args.task)
    args.dev_file = args.dev_file.format(args.task)
    args.output_dir = args.output_dir.format(args.task)
    args.label_file = args.label_file.format(args.task)
    train_data = BERTDataSet(tokenizer, args.max_len, args.label_file, args.train_file)
    dev_data = BERTDataSet(tokenizer, args.max_len, args.label_file, args.dev_file)
    args.label_number = train_data.label_number

    model = BERTClassificationModel(args)

    train(args, model, device, train_data, dev_data)


if __name__ == '__main__':
    main()
