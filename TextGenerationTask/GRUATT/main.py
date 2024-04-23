import os

from model import GRUATTSeq2Seq
from data_set import GRUATTDataSet
from train import train
import argparse
import random
import torch
import numpy as np


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--src_lang', default='en', type=str, help='source language')
    parser.add_argument('--tgt_lang', default='de', type=str, help='target language')
    parser.add_argument('--vocab_file', default='temp_data/vocab', type=str, help='')
    parser.add_argument('--src_emb_file', default='temp_data/emb.json', type=str, help='')
    parser.add_argument('--tgt_emb_file', default='temp_data/emb.json', type=str, help='')
    parser.add_argument('--train_file', default='temp_data/train', type=str, help='')
    parser.add_argument('--dev_file', default='temp_data/dev', type=str, help='')
    parser.add_argument('--src_vocab_size', default=10, type=int, help='')
    parser.add_argument('--tgt_vocab_size', default=10, type=int, help='')
    parser.add_argument('--vec_size', default=100, type=int, help='')
    parser.add_argument('--encoder_hidden_size', default=128, type=int, help='')
    parser.add_argument('--decoder_hidden_size', default=128, type=int, help='')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='')
    parser.add_argument('--num_train_epochs', default=20, type=int, help='')
    parser.add_argument('--train_batch_size', default=128, type=int, help='')
    parser.add_argument('--dev_batch_size', default=128, type=int, help='')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='')
    parser.add_argument('--output_dir', default='output_dir/', type=str, help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--src_max_len', type=int, default=128, help='')
    parser.add_argument('--tgt_max_len', type=int, default=128, help='')
    parser.add_argument('--teacher_forcing', type=float, default=0.5, help='')
    return parser.parse_args()


def main():
    args = set_args()
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    train_data = GRUATTDataSet(args.src_lang, args.tgt_lang, args.vocab_file, args.src_max_len, args.tgt_max_len, args.train_file)
    dev_data = GRUATTDataSet(args.src_lang, args.tgt_lang, args.vocab_file, args.src_max_len, args.tgt_max_len, args.dev_file)
    args.src_vocab_size = train_data.src_vocab_size
    args.tgt_vocab_size = train_data.tgt_vocab_size
    args.start_ids = train_data.tgt_vocab["<bos>"]
    model = GRUATTSeq2Seq(args, src_emb_file=args.src_emb_file, tgt_emb_file=args.tgt_emb_file)

    train(args, model, device, train_data, dev_data)


if __name__ == '__main__':
    main()
