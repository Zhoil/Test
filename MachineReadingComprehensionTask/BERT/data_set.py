import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import logging
import json
from transformers.models.bert.tokenization_bert import whitespace_tokenize
import numpy as np

logger = logging.getLogger(__name__)


class BERTQADataSet(Dataset):
    def __init__(self, tokenizer, max_len, query_max_len, path_file):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.query_max_len = query_max_len
        self.data_set = self.convert_data_to_ids(path_file)

    def convert_data_to_ids(self, path_file):
        self.data_set = []
        with open(path_file, "r", encoding="utf-8") as fh:
            input_data = json.load(fh)["data"]
        for i, entry in enumerate(tqdm(input_data, desc="iter", disable=False)):
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        logger.warning("Could not find answer: {} vs. {}".format(actual_text, cleaned_answer_text))
                        continue
                    sample = {"qas_id": qas_id, "question_text": question_text, "paragraph_text": paragraph_text,
                              "doc_tokens": doc_tokens, "start_position": start_position, "end_position": end_position,
                              "answer_texts": [answer["text"] for answer in qa["answers"]]}
                    sample = self.convert_sample_to_id(sample)
                    self.data_set.append(sample)

        return self.data_set

    def convert_sample_to_id(self, sample):
        query_tokens = self.tokenizer.tokenize(sample["question_text"])
        if len(query_tokens) > self.query_max_len:
            query_tokens = query_tokens[:self.query_max_len]
        sample["query_len"] = len(query_tokens)
        orig_to_tok_index = []
        all_doc_tokens = []
        total = 0
        for (i, token) in enumerate(sample["doc_tokens"]):
            sub_tokens = self.tokenizer.tokenize(token)
            sub_idx = []
            for sub_token in sub_tokens:
                sub_idx.append(total)
                all_doc_tokens.append(sub_token)
                total += 1
            orig_to_tok_index.append(sub_idx)

        max_tokens_for_doc = self.max_len - len(query_tokens) - 3
        end_position = sample["end_position"]
        if len(all_doc_tokens) > max_tokens_for_doc:
            if orig_to_tok_index[end_position][-1] < max_tokens_for_doc:
                all_doc_tokens = all_doc_tokens[:max_tokens_for_doc]
                sample["start_label"] = orig_to_tok_index[sample["start_position"]][0]
                sample["end_label"] = orig_to_tok_index[end_position][-1]
            else:
                start = orig_to_tok_index[end_position][-1] + 1 - max_tokens_for_doc
                all_doc_tokens = all_doc_tokens[start:orig_to_tok_index[sample["end_position"]][-1] + 1]
                sample["start_label"] = orig_to_tok_index[sample["start_position"]][0] - start
                sample["end_label"] = orig_to_tok_index[end_position][-1] - start
        else:
            sample["start_label"] = orig_to_tok_index[sample["start_position"]][0]
            sample["end_label"] = orig_to_tok_index[end_position][-1]
        sample["doc_tokens"] = all_doc_tokens
        tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + all_doc_tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0] * (len(query_tokens) + 2) + [1] * (len(all_doc_tokens) + 1)
        attention_mask = [1] * len(input_ids)
        padding_id = self.tokenizer.convert_tokens_to_ids(['[PAD]'])
        input_ids = input_ids + padding_id * (self.max_len - len(input_ids))
        attention_mask = attention_mask + [0] * (self.max_len - len(attention_mask))
        token_type_ids = token_type_ids + [0] * (self.max_len - len(token_type_ids))
        position_ids = list(np.arange(len(input_ids)))
        sample["input_ids"] = input_ids
        sample["attention_mask"] = attention_mask
        sample["token_type_ids"] = token_type_ids
        sample["position_ids"] = position_ids
        sample["start_label"] = sample["start_label"] + sample["query_len"] + 2
        sample["end_label"] = sample["end_label"] + sample["query_len"] + 2
        return sample

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def collate_func(batch_data):
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    input_ids_list, attention_mask_list, token_type_ids_list, position_ids_list, start_labels_list, end_labels_list = [], [], [], [], [], []
    sample_list = []
    for instance in batch_data:
        input_ids_list.append(instance["input_ids"])
        attention_mask_list.append(instance["attention_mask"])
        token_type_ids_list.append(instance["token_type_ids"])
        position_ids_list.append(instance["position_ids"])
        start_labels_list.append(instance["start_label"])
        end_labels_list.append(instance["end_label"])
        sample_list.append(instance)
    return {"input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids_list, dtype=torch.long),
            "position_ids": torch.tensor(position_ids_list, dtype=torch.long),
            "start_labels": torch.tensor(start_labels_list, dtype=torch.long),
            "end_labels": torch.tensor(end_labels_list, dtype=torch.long),
            "sample": sample_list}
