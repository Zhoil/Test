import os
from model import BiDAFModel
from data_set import BiDAFDataSet, collate_func
from torch.utils.data import DataLoader
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from evaluate import metric_max_over_ground_truths, exact_match_score, f1_score
from train import find_best_answer


def predict(args, model, device, test_data):
    fin = open(args.save_file, "w", encoding="utf-8")  # 打开一个文件args.save_file以写入模型的预测结果
    test_data_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collate_func)
    # 将测试数据划分为大小为args.test_batch_size的批次，并提供一个collate_func用于对批次进行处理。使用tqdm库显示循环进度条，迭代每个批次
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    # 使用tqdm库显示循环进度条，迭代每个批次
    f1, em, total = 0.0, 0.0, 0.0
    out = []
    for step, batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():  # 关闭梯度计 将批次的数据移动到指定的设备
            context_input_ids = batch["context_input_ids"].to(device)
            context_char_input_ids = batch["context_char_input_ids"].to(device)
            query_input_ids = batch["query_input_ids"].to(device)
            query_char_input_ids = batch["query_char_input_ids"].to(device)

            start_logits, end_logits = model(context_input_ids, query_input_ids, context_char_input_ids,
                                             query_char_input_ids)
            # 获取起始和结束位置的logits
            for sample, start_logit, end_logit in zip(batch["sample"], start_logits.cpu().numpy(),
                                                      end_logits.cpu().numpy()):
                best_answer = find_best_answer(sample, start_logit, end_logit)
                # 过find_best_answer函数找到最佳答案，并根据最佳答案计算F1分数和精确匹配得分
                f1 += metric_max_over_ground_truths(f1_score, best_answer, sample["answer_texts"])
                em += metric_max_over_ground_truths(exact_match_score, best_answer, sample["answer_texts"])
                out.append(
                    {"qas_id": sample["qas_id"], "best_answer": best_answer, "answer_texts": sample["answer_texts"],
                     "context": sample["context_tokens"], "question": sample["question_tokens"],
                     "f1": float(metric_max_over_ground_truths(f1_score, best_answer, sample["answer_texts"])),
                     "em": float(
                         metric_max_over_ground_truths(exact_match_score, best_answer, sample["answer_texts"]))})
                # 将结果添加到out列表中，其中包括问题id、最佳答案、基准答案、上下文、问题、F1分数和精确匹配得分
            total += len(batch["context_input_ids"])  # 累加总次数total，计算总的F1分数和精确匹配得分
    eval_f1 = f1 / total
    eval_em = em / total
    json.dump(out, fin, ensure_ascii=False, indent=4)  # 将预测结果以JSON格式写入文件
    fin.close()  # 关闭文件
    return eval_em, eval_f1  # 将平均精确匹配得分和F1分数作为结果返回


def set_args():  # 同Classifition
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--word_vocab_file', default='temp_data/word_vocab.json', type=str, help='')
    parser.add_argument('--char_vocab_file', default='temp_data/char_vocab.json', type=str, help='')
    parser.add_argument('--emb_file', default='temp_data/emb.json', type=str, help='')
    parser.add_argument('--word_vocab_size', default=10, type=int, help='')
    parser.add_argument('--char_vocab_size', default=10, type=int, help='')
    parser.add_argument('--word_vec_size', default=100, type=int, help='')
    parser.add_argument('--char_vec_size', default=25, type=int, help='')
    parser.add_argument('--hidden_size', default=100, type=int, help='')
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='')
    parser.add_argument('--test_batch_size', default=16, type=int, help='')
    parser.add_argument('--context_max_len', type=int, default=512, help='')
    parser.add_argument('--query_max_len', type=int, default=64, help='')
    parser.add_argument('--word_max_len', type=int, default=8, help='')
    parser.add_argument('--model_path', type=str, default="output_dir/checkpoint-epoch1-bs-8-lr-0.001", help='')
    parser.add_argument('--test_file', type=str, default="../data/dev-v1.1.json", help='')
    parser.add_argument('--save_file', type=str, default="temp_data/result.txt", help='')
    return parser.parse_args()


def main():  # 同Classifition
    args = set_args()
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    test_data = BiDAFDataSet(args.word_vocab_file, args.char_vocab_file, args.context_max_len, args.query_max_len,
                             args.word_max_len, args.test_file)
    args.word_vocab_size = test_data.word_vocab_size
    args.char_vocab_size = test_data.char_vocab_size
    model = BiDAFModel(args, args.emb_file)
    model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt"), map_location="cpu"))
    model.eval()
    model.to(device)
    em, f1 = predict(args, model, device, test_data)
    print("测试数据的精准率为{}，f1为{}".format(em, f1))


if __name__ == '__main__':
    main()
