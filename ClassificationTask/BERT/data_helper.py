from tqdm import tqdm
import json
import argparse

def build_vocab(train_file, dev_file, label_file):
    label_dict = {}
    for path in [train_file, dev_file]:
        with open(path, "r", encoding="utf-8") as fh:
            for _, line in enumerate(tqdm(fh, desc="iter", disable=False)):
                line = line.rstrip().split("\t")
                if line[-1] not in label_dict:
                    label_dict[line[-1]] = len(label_dict)
    with open(label_file, "w", encoding="utf-8") as fh:
        json.dump(label_dict, fh, ensure_ascii=False, indent=4)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='SNIPS', type=str, help='')
    parser.add_argument('--train_file', default='../data/{}/train.tsv', type=str, help='')
    parser.add_argument('--dev_file', default='../data/{}/valid.tsv', type=str, help='')
    parser.add_argument('--label_file', default='temp_data/{}/label.json', type=str, help='')
    return parser.parse_args()

if __name__ == '__main__':
    args = set_args()
    args.train_file = args.train_file.format(args.task)
    args.dev_file = args.dev_file.format(args.task)
    args.label_file = args.label_file.format(args.task)
    build_vocab(args.train_file, args.dev_file, args.label_file)
