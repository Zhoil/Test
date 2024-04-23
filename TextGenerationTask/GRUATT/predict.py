import os
from model import GRUATTSeq2Seq
from data_set import GRUATTDataSet, collate_func
from torch.utils.data import DataLoader
import torch
import argparse
import json
from tqdm import tqdm
from train import get_bleu


def predict(args, model, device, test_data):
    data = []
    test_data_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collate_func)
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    vocab = test_data.tgt_vocab
    id2dict = {v: k for k, v in vocab.items()}
    sys = []
    refs = []
    for step, batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            input_ids = batch["src_input_ids"].to(device)
            samples = batch["bitext"]
            outputs = model.forward(input_ids=input_ids)[0]
            for sample, t_o in zip(samples, outputs):
                pre_target = [id2dict[int(l)] for l in t_o.cpu().numpy().tolist()]
                pre_target = " ".join(pre_target[1:]).split("<eos>")[0]
                sys.append(pre_target)
                refs.append(sample["tgt_text"])
                data.append({"source": sample["src_text"], "target": sample["tgt_text"], "pre_target": pre_target})
    sacrebleu_result = get_bleu(sys, refs)
    fin = open(args.save_file, "w", encoding="utf-8")
    json.dump(data, fin, ensure_ascii=False, indent=4)
    fin.close()
    return sacrebleu_result


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--src_lang', default='en', type=str, help='source language')
    parser.add_argument('--tgt_lang', default='de', type=str, help='target language')
    parser.add_argument('--vocab_file', default='temp_data/vocab', type=str, help='')
    parser.add_argument('--src_emb_file', default='temp_data/emb.json', type=str, help='')
    parser.add_argument('--tgt_emb_file', default='temp_data/emb.json', type=str, help='')
    parser.add_argument('--src_vocab_size', default=10, type=int, help='')
    parser.add_argument('--tgt_vocab_size', default=10, type=int, help='')
    parser.add_argument('--vec_size', default=100, type=int, help='')
    parser.add_argument('--encoder_hidden_size', default=128, type=int, help='')
    parser.add_argument('--decoder_hidden_size', default=128, type=int, help='')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='')
    parser.add_argument('--test_batch_size', default=64, type=int, help='')
    parser.add_argument('--src_max_len', type=int, default=512, help='')
    parser.add_argument('--tgt_max_len', type=int, default=512, help='')
    parser.add_argument('--model_path', type=str, default="output_dir/checkpoint-epoch19-bs-128-lr-0.001", help='')
    parser.add_argument('--test_file', type=str, default="temp_data/test", help='')
    parser.add_argument('--save_file', type=str, default="temp_data/result.json", help='')
    return parser.parse_args()


def main():
    args = set_args()
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    test_data = GRUATTDataSet(args.src_lang, args.tgt_lang, args.vocab_file, args.src_max_len, args.tgt_max_len, args.test_file)
    args.src_vocab_size = test_data.src_vocab_size
    args.tgt_vocab_size = test_data.tgt_vocab_size
    args.start_ids = test_data.tgt_vocab["<bos>"]
    model = GRUATTSeq2Seq(args)
    model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt"), map_location="cpu"))
    model.eval()
    model.to(device)
    sacrebleu_result = predict(args, model, device, test_data)
    print("测试数据的结果为: {}".format(sacrebleu_result.format()))


if __name__ == '__main__':
    main()
