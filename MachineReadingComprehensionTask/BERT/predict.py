import os
from model import BERTQAModel
from transformers import BertTokenizer
from data_set import BERTQADataSet, collate_func
from torch.utils.data import DataLoader
import torch
import argparse
from tqdm import tqdm
import json
from train import find_best_answer
from evaluate import exact_match_score, f1_score, metric_max_over_ground_truths


def predict(args, model, device, test_data):
    model.to(device)
    test_data_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collate_func)
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    f1, em, total = 0.0, 0.0, 0.0
    out = []
    for step, batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            start_logits, end_logits = model(input_ids=input_ids, attention_mask=attention_mask,
                                             token_type_ids=token_type_ids, position_ids=position_ids)
            for sample, start_logit, end_logit in zip(batch["sample"], start_logits.cpu().numpy(),
                                                      end_logits.cpu().numpy()):
                best_answer = find_best_answer(sample, start_logit, end_logit)
                f1_ = metric_max_over_ground_truths(f1_score, best_answer, sample["answer_texts"])
                f1 += f1_
                em_ = metric_max_over_ground_truths(exact_match_score, best_answer, sample["answer_texts"])
                em += em_
                out.append(
                    {"qas_id": sample["qas_id"], "best_answer": best_answer, "question_text": sample["question_text"],
                     "paragraph_text": sample["paragraph_text"], "answer_texts": sample["answer_texts"],
                     "f1": float(f1_), "em": float(em_)})
            total += len(batch["input_ids"])
    test_f1 = f1 / total
    test_em = em / total
    fin = open(args.save_file, "w", encoding="utf-8")
    json.dump(out, fin, ensure_ascii=False, indent=4)
    fin.close()
    return test_em, test_f1


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--vocab_path', default='../data/bert_base/vocab.txt', type=str, help='')
    parser.add_argument('--max_len', type=int, default=512, help='')
    parser.add_argument('--query_max_len', type=int, default=128, help='')
    parser.add_argument('--hidden_size', default=768, type=int, help='')
    parser.add_argument('--test_batch_size', type=int, default=8, help='')
    parser.add_argument('--pre_train_model', default='../data/bert_base/', type=str, help='')
    parser.add_argument('--model_path', type=str, default="output_dir/checkpoint-epoch3-bs-4-lr-2e-05", help='')
    parser.add_argument('--test_file', type=str, default="../data/dev-v1.1.json", help='')
    parser.add_argument('--save_file', type=str, default="temp_data/result.json", help='')
    return parser.parse_args()


def main():
    args = set_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    test_data = BERTQADataSet(tokenizer, args.max_len, args.query_max_len, args.test_file)
    model = BERTQAModel(args)
    model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt"), map_location="cpu"))
    model.eval()
    em, f1 = predict(args, model, device, test_data)
    print("测试数据的精准率为{}，f1为{}".format(em, f1))


if __name__ == '__main__':
    main()
