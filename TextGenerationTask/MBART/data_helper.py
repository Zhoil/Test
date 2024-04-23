import re
import json
import argparse
from tqdm import tqdm


def data_cut(file, save_file, sample_nums=20000):
    fin = open(save_file, "w", encoding="utf-8")
    with open(file, "r", encoding="utf-8") as fr:
        lines = fr.readlines()
        sample_lines = lines[:sample_nums]
        for i, line in enumerate(tqdm(sample_lines)):
            fin.write(line)
    fin.close()


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sample-nums", type=int, default=20000)
    parser.add_argument("--src-lang", type=str, default="en")
    parser.add_argument("--tgt-lang", type=str, default="de")
    return parser.parse_args()

if __name__ == '__main__':
    args = set_args()
    langs = [args.src_lang, args.tgt_lang]
    ori_path = ["../data/raw/train.{}", "../data/raw/valid.{}", "../data/raw/test.{}"]
    save_path = ["temp_data/train.{}", "temp_data/dev.{}", "temp_data/test.{}"]
    sample_nums = [args.train_sample_nums, None, None]

    for lang in langs:
        for o_p, s_p, s_n in zip(ori_path, save_path, sample_nums):
            data_cut(file=o_p.format(lang), save_file=s_p.format(lang), sample_nums=s_n)
