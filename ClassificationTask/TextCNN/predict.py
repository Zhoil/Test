import os
from model import TextCNN
from data_set import TextCNNDataSet, collate_func
from torch.utils.data import DataLoader
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm

def predict(args, model, device, test_data):  # 在给定模型和测试数据上进行预测
    fin = open(args.save_file, "w", encoding="utf-8")  # 打开一个输出文件，用于保存预测结果

    test_data_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collate_func)  # 创建一个数据加载器
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    y_true = []
    y_predict = []
    data = []
    total_loss, total = 0.0, 0.0
    for step, batch in enumerate(iter_bar):  # 使用循环逐批次地进行预测
        model.eval()
        with torch.no_grad():  # 禁用梯度计算
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            y_true.extend(labels.cpu().numpy().tolist())
            data.extend(batch["text"])
            outputs = model.forward(input_ids=input_ids, labels=labels)
            y_label = torch.argmax(outputs[1], dim=-1)
            # score = outputs[1][:, 1].cpu().numpy()
            y_predict.extend(y_label.cpu().numpy().tolist())  # 将预测结果和真实标签保存到列表中

    label2str_dict = dict((v, k) for k, v in test_data.label_dict.items())
    for i, line in enumerate(tqdm(zip(data, y_true, y_predict), desc="iter", disable=False)):  # 将预测结果写入输出文件
        fin.write("{}\t{}\t{}\n".format(line[0], label2str_dict[line[1]], label2str_dict[line[2]]))
    fin.close()

    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    test_acc = np.mean((y_true == y_predict))  # 计算测试准确率
    return test_acc  # 计算并返回测试准确率


def set_args():  # 同
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--task', default='SNIPS', type=str, help='')
    parser.add_argument('--vocab_file', default='temp_data/vocab.json', type=str, help='')
    parser.add_argument('--label_file', default='temp_data/label.json', type=str, help='')
    parser.add_argument('--vocab_size', default=10, type=int, help='')
    parser.add_argument('--test_batch_size', default=8, type=int, help='')
    parser.add_argument('--vec_size', default=100, type=int, help='')
    parser.add_argument('--filter_num', default=100, type=int, help='')
    parser.add_argument('--in_channels', default=1, type=int, help='')
    parser.add_argument('--kernels', default=[3, 4, 5], type=list, help='')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='')
    parser.add_argument('--max_len', type=int, default=128, help='')
    parser.add_argument('--label_number', default=2, type=int, help='')
    parser.add_argument('--model_path', type=str, default="output_dir/SST2/checkpoint-epoch4-bs-50-lr-0.0005", help='')
    parser.add_argument('--test_file', type=str, default="../data/{}/test.tsv", help='')
    parser.add_argument('--save_file', type=str, default="temp_data/{}/result.txt", help='')
    return parser.parse_args()


def main():
    args = set_args()
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    args.test_file = args.test_file.format(args.task)
    args.save_file = args.save_file.format(args.task)
    test_data = TextCNNDataSet(args.vocab_file, args.max_len, args.label_file, args.test_file)
    args.vocab_size = test_data.vocab_size
    args.label_number = test_data.label_number

    model = TextCNN(args)
    model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt"), map_location="cpu"))
    model.eval()
    model.to(device)
    acc = predict(args, model, device, test_data)
    print("{}任务上测试数据的准确率为{}".format(args.task, acc))


if __name__ == '__main__':
    main()
