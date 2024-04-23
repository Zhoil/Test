import torch
import os

import logging
from data_set import collate_func
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
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
    max_bleu = 0.
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        model.train()
        for step, batch in enumerate(iter_bar):
            input_ids = batch["src_input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_ids = batch["tgt_label_ids"].to(device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)[0]
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        sacrebleu_result = evaluate(args, model, device, dev_data)
        bleu_score = sacrebleu_result.score
        tb_write.add_scalar("bleu", bleu_score, i_epoch)
        tb_write.add_scalar("bleu_1", sacrebleu_result.precisions[0], i_epoch)
        tb_write.add_scalar("bleu_2", sacrebleu_result.precisions[1], i_epoch)
        tb_write.add_scalar("bleu_3", sacrebleu_result.precisions[2], i_epoch)
        tb_write.add_scalar("bleu_4", sacrebleu_result.precisions[3], i_epoch)
        logging.info(
            "i_epoch is {}, {}".format(i_epoch, sacrebleu_result.format()))
        if bleu_score > max_bleu:
            max_bleu = bleu_score 
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
    bos_token_id = dev_data.bos_token_id
    pad_token_id = dev_data.pad_token_id
    eos_token_id = dev_data.eos_token_id
    tokenizer = dev_data.tokenizer
    sys = []
    refs = []
    for step, batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            input_ids = batch["src_input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bitexts = batch["bitext"]
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=args.tgt_max_len, bos_token_id=eos_token_id,
                                     pad_token_id=pad_token_id, eos_token_id=eos_token_id, num_return_sequences=1,
                                     num_beams=1, do_sample=False, num_beam_groups=1)
            pre_targets = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            sys.extend(pre_targets)
            for bitext in bitexts:
                refs.append(bitext["tgt_text"])
    sacrebleu_result = get_bleu(sys, refs)

    return sacrebleu_result


def get_bleu(sys, refs):
    import sacrebleu
    sacrebleu_result = sacrebleu.corpus_bleu(
            sys, [refs], tokenize="none"
        )
    return sacrebleu_result