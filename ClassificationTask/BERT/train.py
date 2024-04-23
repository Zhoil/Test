import os
from data_set import BERTDataSet, collate_func
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch
import logging
import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args, model, device, train_data, dev_data):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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
    max_acc = 0.
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        model.train()
        for step, batch in enumerate(iter_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            labels = batch["labels"].to(device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                         position_ids=position_ids, labels=labels)[0]
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        eval_acc, eval_loss = evaluate(args, model, device, dev_data)
        tb_write.add_scalar("dev_acc", eval_acc, i_epoch)
        tb_write.add_scalar("dev_loss", eval_loss, i_epoch)
        logging.info("i_epoch is {}, dev_acc is {}, dev_loss is {}".format(i_epoch, eval_acc, eval_loss))
        if eval_acc > max_acc:
            max_acc = eval_acc
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
    y_true = []
    y_predict = []
    total_loss, total = 0.0, 0.0
    for step, batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            labels = batch["labels"].to(device)
            y_true.extend(labels.cpu().numpy().tolist())
            outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                    position_ids=position_ids, labels=labels)
            # score = outputs[1][:, 1]
            y_label = torch.argmax(outputs[1], dim=-1)
            y_predict.extend(y_label.cpu().numpy().tolist())
            loss = outputs[0].item()
            total_loss += loss * len(batch["input_ids"])
            total += len(batch["input_ids"])
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    eval_acc = np.mean((y_true == y_predict))
    eval_loss = total_loss / total
    return eval_acc, eval_loss
