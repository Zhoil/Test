import json
from tqdm import tqdm
import numpy as np
import os
import argparse


def data_cut(file, save_file, sample_nums=20000):
    fin = open(save_file, "w", encoding="utf-8")
    with open(file, "r", encoding="utf-8") as fr:
        lines = fr.readlines()
        sample_lines = lines[:sample_nums]
        for i, line in enumerate(tqdm(sample_lines)):
            fin.write(line)
    fin.close()


def build_vocab(train_file, dev_file, vocab_file):
    vocab = {'<PAD>': 0, '<UNK>': 1, '<bos>': 2, '<eos>': 3}
    for path in [train_file, dev_file]:
        with open(path, "r", encoding="utf-8") as fh:
            for _, line in enumerate(tqdm(fh, desc="iter", disable=False)):
                sample = line.strip().split()
                for token in sample:
                    if token not in vocab:
                        vocab[token] = len(vocab)
                    else:
                        continue
    with open(vocab_file, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=4)
    return vocab


def load_emb(w2v_file, vocab, emb_file):
    w2v_dict = {}
    with open(w2v_file, "r", encoding="utf-8") as fh:
        for i, line in enumerate(tqdm(fh, desc="iter", disable=False)):
            array = line.rstrip().split(' ')
            word, vector = array[0], array[1:]
            vec_size = len(array[1:])
            w2v_dict[word] = list(map(float, vector))
    emb_dict = {}
    for k, v in vocab.items():
        if k in w2v_dict:
            emb_dict[k] = w2v_dict[k]
        else:
            emb_dict[k] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
    idx2emb_dict = {idx: emb_dict[word] for word, idx in vocab.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    fin = open(emb_file, "w", encoding="utf-8")
    json.dump(emb_mat, fin)
    fin.close()


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sample-nums", type=int, default=20000)
    parser.add_argument("--token-type", type=str, default="bpe")
    parser.add_argument("--src-lang", type=str, default="en")
    parser.add_argument("--tgt-lang", type=str, default="de")
    parser.add_argument("--src-word-vector-file", type=str, default=None)
    parser.add_argument("--tgt-word-vector-file", type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = set_args()
    langs = [args.src_lang, args.tgt_lang]
    word_vector_files = [args.src_word_vector_file, args.tgt_word_vector_file]
    ori_path = ["../data/{}/train.{}", "../data/{}/valid.{}", "../data/{}/test.{}"]
    save_path = ["temp_data/train.{}", "temp_data/dev.{}", "temp_data/test.{}"]
    sample_nums = [args.train_sample_nums, None, None]
    for lang, word_vector_file in zip(langs, word_vector_files):
        for o_p, s_p, s_n in zip(ori_path, save_path, sample_nums):
            data_cut(o_p.format(args.token_type, lang), s_p.format(lang), s_n)

        vocab_file_path = "temp_data/vocab.{}.json".format(lang)
        vocab = build_vocab(save_path[0].format(lang), save_path[1].format(lang), vocab_file_path)
       
        if word_vector_file is not None and os.path.exists(word_vector_file):
            emb_file_path = "temp_data/emb.{}.json".format(lang)
            load_emb(word_vector_file, vocab, emb_file_path)
