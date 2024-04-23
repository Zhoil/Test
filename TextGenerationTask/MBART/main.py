import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from train import train
from data_set import MBARTDataSet 
import torch
import argparse
import random
import numpy as np


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--src_lang', default='en', type=str, help='source language')
    parser.add_argument('--tgt_lang', default='de', type=str, help='target language')
    parser.add_argument('--mbart_src_lang_code', default='en_XX', type=str, help='mBART source language code')
    parser.add_argument('--mbart_tgt_lang_code', default='de_DE', type=str, help='mBART target language code')
    parser.add_argument('--pre_train_model', default='../data/mbart_large_50', type=str, help='')
    parser.add_argument('--train_file', default='temp_data/train', type=str, help='')
    parser.add_argument('--dev_file', default='temp_data/dev', type=str, help='')
    parser.add_argument('--num_train_epochs', default=20, type=int, help='')
    parser.add_argument('--train_batch_size', default=16, type=int, help='')
    parser.add_argument('--dev_batch_size', default=24, type=int, help='')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='')
    parser.add_argument('--output_dir', default='output_dir/', type=str, help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--src_max_len', type=int, default=64, help='')
    parser.add_argument('--tgt_max_len', type=int, default=64, help='')
    return parser.parse_args()


def main():
    args = set_args()
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)


    tokenizer = MBart50TokenizerFast.from_pretrained(args.pre_train_model, src_lang=args.mbart_src_lang_code, tgt_lang=args.mbart_tgt_lang_code)
    model = MBartForConditionalGeneration.from_pretrained(args.pre_train_model)
    train_data = MBARTDataSet(tokenizer, args.src_lang, args.tgt_lang, args.src_max_len, args.tgt_max_len, args.train_file)
    dev_data = MBARTDataSet(tokenizer, args.src_lang, args.tgt_lang, args.src_max_len, args.tgt_max_len, args.dev_file)


    train(args, model, device, train_data, dev_data)


if __name__ == '__main__':
    main()
