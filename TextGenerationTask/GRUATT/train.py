import os
from data_set import collate_func
from torch.utils.data import DataLoader
import torch
import logging
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
    model.to(device)
    paras = model.parameters()
    optimizer = torch.optim.Adam(paras, lr=args.learning_rate)
    model.zero_grad()
    max_bleu = 0.

    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        model.train()
        for step, batch in enumerate(iter_bar):
            input_ids = batch["src_input_ids"].to(device)
            labels = batch["shifted_tgt_ids"].to(device)
            loss = model(input_ids, labels)[0]
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            loss.backward()
            optimizer.step()
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
    vocab = dev_data.tgt_vocab
    id2dict = {v: k for k, v in vocab.items()}
    sys = []
    refs = []
    for step, batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            input_ids = batch["src_input_ids"].to(device)
            text = batch["bitext"]
            outputs = model.forward(input_ids=input_ids)[0]
            for t, t_o in zip(text, outputs):
                pre_target = [id2dict[int(l)] for l in t_o.cpu().numpy().tolist()]
                pre_target = " ".join(pre_target[1:]).split("<eos>")[0]
                sys.append(pre_target)
                refs.append(t["tgt_text"])
    sacrebleu_result = get_bleu(sys, refs)

    return sacrebleu_result


def get_bleu(sys, refs):
    import sacrebleu
    sacrebleu_result = sacrebleu.corpus_bleu(
            sys, [refs], tokenize="none"
        )
    return sacrebleu_result
