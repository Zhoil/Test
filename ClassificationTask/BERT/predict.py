import os
from model import BERTClassificationModel
from transformers import BertTokenizer
from data_set import BERTDataSet, collate_func
from torch.utils.data import DataLoader
import torch
import argparse
from tqdm import tqdm
import json
import numpy as np


def predict(args, model, device, test_data):
    model.to(device)
    fin = open(args.save_file, "w", encoding="utf-8")
    test_data_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collate_func)
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    y_true = []
    y_predict = []
    data = []
    for step, batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            labels = batch["labels"].to(device)
            data.extend(batch["text"])
            y_true.extend(labels.cpu().numpy().tolist())
            outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                    position_ids=position_ids, labels=labels)
            y_label = torch.argmax(outputs[1], dim=-1)
            # score = outputs[1][:, 1].cpu().numpy()
            y_predict.extend(y_label.cpu().numpy().tolist())

    label2str_dict = dict((v, k) for k, v in test_data.label_dict.items())
    for i, line in enumerate(tqdm(zip(data, y_true, y_predict), desc="iter", disable=False)):
        fin.write("{}\t{}\t{}\n".format(line[0], label2str_dict[line[1]], label2str_dict[line[2]]))
    fin.close()
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    test_acc = np.mean((y_true == y_predict))
    return test_acc


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--task', default='SNIPS', type=str, help='')
    parser.add_argument('--vocab_path', default='../data/bert_base/vocab.txt', type=str, help='')
    parser.add_argument('--max_len', type=int, default=128, help='')
    parser.add_argument('--hidden_size', default=768, type=int, help='')
    parser.add_argument('--label_number', type=int, default=2, help='')
    parser.add_argument('--test_batch_size', type=int, default=8, help='')
    parser.add_argument('--pre_train_model', default='../data/bert_base/', type=str, help='')
    parser.add_argument('--label_file', default='temp_data/{}/label.json', type=str, help='')
    # parser.add_argument('--model_path', default='../data/bert_base/', type=str, help='')
    parser.add_argument('--model_path', type=str, default="output_dir/SNIPS/checkpoint-epoch3-bs-8-lr-2e-05", help='')
    parser.add_argument('--test_file', type=str, default="../data/{}/test.tsv", help='')
    parser.add_argument('--save_file', type=str, default="temp_data/{}/result.txt", help='')
    return parser.parse_args()


def main():
    args = set_args()
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    
    args.test_file = args.test_file.format(args.task)
    args.save_file = args.save_file.format(args.task)
    args.label_file = args.label_file.format(args.task)
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    test_data = BERTDataSet(tokenizer, args.max_len, args.label_file, args.test_file)
    args.label_number = test_data.label_number
    model = BERTClassificationModel(args)
    model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt"), map_location="cpu"))
    model.eval()
    acc = predict(args, model, device, test_data)
    print("测试数据的准确率为{}".format(acc))


if __name__ == '__main__':
    main()
