import os
from data_set import collate_func
from torch.utils.data import DataLoader
import torch
import logging
import numpy as np
from tqdm import tqdm, trange
from evaluate import metric_max_over_ground_truths, exact_match_score, f1_score
import json
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args, model, device, train_data, dev_data):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    tb_write = SummaryWriter()
    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.train_batch_size,
                              collate_fn=collate_func,
                              shuffle=True)
    model.to(device)
    paras = model.parameters()
    optimizer = torch.optim.Adam(paras, lr=args.learning_rate)
    model.zero_grad()
    max_f1 = 0.
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        model.train()
        for step, batch in enumerate(iter_bar):
            context_input_ids = batch["context_input_ids"].to(device)
            context_char_input_ids = batch["context_char_input_ids"].to(device)
            query_input_ids = batch["query_input_ids"].to(device)
            query_char_input_ids = batch["query_char_input_ids"].to(device)
            start_labels = batch["start_labels"].to(device)
            end_labels = batch["end_labels"].to(device)
            loss = model(context_input_ids, query_input_ids, context_char_input_ids, query_char_input_ids, start_labels,
                         end_labels)[0]
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        eval_em, eval_f1 = evaluate(args, model, device, dev_data)
        tb_write.add_scalar("dev_em", eval_em, i_epoch)
        tb_write.add_scalar("dev_f1", eval_f1, i_epoch)
        logging.info(
            "epoch is {}, dev_em is {}, dev_f1 is {}".format(i_epoch, eval_em, eval_f1))
        if eval_f1 > max_f1:
            max_f1 = eval_f1
            output_dir = os.path.join(args.output_dir,
                                      'checkpoint-epoch{}-bs-{}-lr-{}'.format(i_epoch, args.train_batch_size,
                                                                              args.learning_rate))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))
        torch.cuda.empty_cache()
    logger.info('Train done')


def evaluate(args, model, device, dev_data):
    dev_data_loader = DataLoader(dev_data, batch_size=args.dev_batch_size, collate_fn=collate_func)
    iter_bar = tqdm(dev_data_loader, desc="iter", disable=False)
    f1, em, total = 0.0, 0.0, 0.0
    out = []
    for step, batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            context_input_ids = batch["context_input_ids"].to(device)
            context_char_input_ids = batch["context_char_input_ids"].to(device)
            query_input_ids = batch["query_input_ids"].to(device)
            query_char_input_ids = batch["query_char_input_ids"].to(device)
            start_logits, end_logits = model(context_input_ids, query_input_ids, context_char_input_ids,
                                             query_char_input_ids)
            for sample, start_logit, end_logit in zip(batch["sample"], start_logits.cpu().numpy(),
                                                      end_logits.cpu().numpy()):
                best_answer = find_best_answer(sample, start_logit, end_logit)
                f1 += metric_max_over_ground_truths(f1_score, best_answer, sample["answer_texts"])
                em += metric_max_over_ground_truths(exact_match_score, best_answer, sample["answer_texts"])
                out.append(
                    {"qas_id": sample["qas_id"], "best_answer": best_answer, "answer_texts": sample["answer_texts"],
                     "context": sample["context_tokens"], "question": sample["question_tokens"],
                     "f1": float(metric_max_over_ground_truths(f1_score, best_answer, sample["answer_texts"])),
                     "em": float(
                         metric_max_over_ground_truths(exact_match_score, best_answer, sample["answer_texts"]))})
            total += len(batch["context_input_ids"])
    eval_f1 = f1 / total
    eval_em = em / total
    return eval_em, eval_f1


def find_best_answer(sample, start_score, end_score):
    context_len = min(start_score.shape[0], len(sample["context_tokens"]))
    score = 0.0
    best_anwser = ""
    for i in range(context_len):
        end_p = np.argmax(end_score[i:])
        end_s = np.max(end_score[i:])
        if float(start_score[i]) * float(end_s) > score and end_p + i <= context_len:
            score = float(start_score[i]) * float(end_s)
            best_anwser = " ".join(sample["context_tokens"][i:i + end_p + 1])
    return best_anwser
