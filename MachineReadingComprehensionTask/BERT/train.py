import os
from data_set import collate_func
from transformers import AdamW, get_linear_schedule_with_warmup
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
    total_steps = int(len(train_loader) * args.num_train_epochs)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    
    max_em, max_f1 = 0., 0.
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        model.train()
        for step, batch in enumerate(iter_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            start_labels = batch["start_labels"].to(device)
            end_labels = batch["end_labels"].to(device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                         position_ids=position_ids, start_labels=start_labels, end_labels=end_labels)[0]
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        eval_em, eval_f1 = evaluate(args, model, device, dev_data)
        tb_write.add_scalar("dev_em", eval_em, i_epoch)
        tb_write.add_scalar("dev_f1", eval_f1, i_epoch)
        logging.info("i_epoch is {}, dev_em is {}, dev_f1 is {}".format(i_epoch, eval_em, eval_f1))
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            start_logits, end_logits = model(input_ids=input_ids, attention_mask=attention_mask,
                                             token_type_ids=token_type_ids, position_ids=position_ids)
            for sample, start_logit, end_logit in zip(batch["sample"], start_logits.cpu().numpy(),
                                                      end_logits.cpu().numpy()):
                best_answer = find_best_answer(sample, start_logit, end_logit)
                f1 += metric_max_over_ground_truths(f1_score, best_answer, sample["answer_texts"])
                em += metric_max_over_ground_truths(exact_match_score, best_answer, sample["answer_texts"])
                out.append(
                    {"qas_id": sample["qas_id"], "best_answer": best_answer, "answer_texts": sample["answer_texts"],
                     "f1": float(metric_max_over_ground_truths(f1_score, best_answer, sample["answer_texts"])),
                     "em": float(
                         metric_max_over_ground_truths(exact_match_score, best_answer, sample["answer_texts"]))})
            total += len(batch["input_ids"])
    eval_f1 = f1 / total
    eval_em = em / total
    return eval_em, eval_f1


def find_best_answer(sample, start_logit, end_logit):
    start_logit[:sample["query_len"] + 2] = np.ones(sample["query_len"] + 2) * -float("inf")
    end_logit[:sample["query_len"] + 2] = np.ones(sample["query_len"] + 2) * -float("inf")
    y_start = np.argmax(start_logit)
    y_end = np.argmax(end_logit)
    if y_start > y_end:
        best_anwser = " ".join(sample["doc_tokens"]).replace(" ##", "").replace("##", "")
    else:
        best_anwser = " ".join(sample["doc_tokens"][y_start - sample["query_len"] - 2:y_end - sample["query_len"] - 1])
        best_anwser = best_anwser.replace(" ##", "").replace("##", "")
    return best_anwser
