import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from data_set import MBARTDataSet, collate_func
from torch.utils.data import DataLoader
import torch
import argparse
from tqdm import tqdm
import json
from train import get_bleu 


def predict(args, model, device, test_data):
    model.to(device)
    data = []
    test_data_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collate_func)
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    bos_token_id = test_data.bos_token_id
    pad_token_id = test_data.pad_token_id
    eos_token_id = test_data.eos_token_id
    tokenizer = test_data.tokenizer
    sys = []
    refs = []
    model.eval()
    for step, batch in enumerate(iter_bar):
        with torch.no_grad():
            input_ids = batch["src_input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bitexts = batch["bitext"]
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=args.tgt_max_len, forced_bos_token_id=test_data.tgt_lang_code_id,
                                     pad_token_id=pad_token_id, eos_token_id=eos_token_id, num_return_sequences=1,
                                     num_beams=1, do_sample=False, num_beam_groups=1)
            pre_targets = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            sys.extend(pre_targets)
            for bitext, pre_target in zip(bitexts, pre_targets):
                refs.append(bitext["tgt_text"])
                data.append({"source": bitext["src_text"], "target": bitext["tgt_text"], "pre_target": pre_target})
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
    parser.add_argument('--mbart_src_lang_code', default='en_XX', type=str, help='mBART source language code')
    parser.add_argument('--mbart_tgt_lang_code', default='de_DE', type=str, help='mBART target language code')
    parser.add_argument('--src_max_len', type=int, default=128, help='')
    parser.add_argument('--tgt_max_len', type=int, default=128, help='')
    parser.add_argument('--test_batch_size', type=int, default=64, help='')
    parser.add_argument('--pre_train_model', default='../data/mbart_large_50', type=str, help='')
    parser.add_argument('--model_path', type=str, default="output_dir/checkpoint-epoch4-bs-16-lr-2e-05", help='')
    parser.add_argument('--test_file', type=str, default="temp_data/test", help='')
    parser.add_argument('--save_file', type=str, default="temp_data/result.json", help='')
    return parser.parse_args()


def main():
    args = set_args()
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    tokenizer = MBart50TokenizerFast.from_pretrained(args.pre_train_model, src_lang=args.mbart_src_lang_code, tgt_lang=args.mbart_tgt_lang_code)
    test_data = MBARTDataSet(tokenizer, args.src_lang, args.tgt_lang, args.src_max_len, args.tgt_max_len, args.test_file)

    model = MBartForConditionalGeneration.from_pretrained(args.pre_train_model)
    model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt"), map_location="cpu"))
    model.eval()
    sacrebleu_result = predict(args, model, device, test_data)
    print("测试数据的结果为: {}".format(sacrebleu_result.format()))


if __name__ == '__main__':
    main()
