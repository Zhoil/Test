import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import json
import random


class Encoder(nn.Module):
    def __init__(self, embedding, vec_size, encoder_hidden_size, decoder_hidden_size, dropout=0.1):
        super().__init__()
        self.embedding = embedding
        self.rnn = nn.GRU(vec_size, encoder_hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        input_emb = self.embedding(input_ids)
        input_emb = self.dropout(input_emb)
        encoder_outputs, encoder_hiddens = self.rnn(input_emb)
        s = torch.tanh(self.fc(torch.cat((encoder_hiddens[-2, :, :], encoder_hiddens[-1, :, :]), dim=1)))
        return encoder_outputs, s


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super().__init__()
        self.attn = nn.Linear((encoder_hidden_size * 2) + decoder_hidden_size, decoder_hidden_size, bias=False)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)

    def forward(self, s, encoder_outputs):
        s = s.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1)
        energy = torch.tanh(self.attn(torch.cat((s, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = F.softmax(attention, dim=1)
        return attention


class Decoder(nn.Module):
    def __init__(self, embedding, vocab_size, vec_size, encoder_hidden_size, decoder_hidden_size, dropout=0.1):
        super().__init__()
        self.embedding = embedding
        self.attention = Attention(encoder_hidden_size, decoder_hidden_size)
        self.rnn = nn.GRU(vec_size, decoder_hidden_size, batch_first=True)
        self.fc_out = nn.Linear((encoder_hidden_size * 2) + decoder_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_input_ids, s, encoder_outputs):
        decoder_input_ids = decoder_input_ids.unsqueeze(1)
        embedded = self.dropout(self.embedding(decoder_input_ids))
        att = self.attention(s, encoder_outputs).unsqueeze(1)
        c = torch.bmm(att, encoder_outputs)
        _, decoder_hidden = self.rnn(embedded, s.unsqueeze(0))
        pred = self.fc_out(torch.cat((decoder_hidden.squeeze(0).squeeze(1), c.squeeze(1)), dim=1))

        return pred, decoder_hidden.squeeze(0)


class GRUATTSeq2Seq(nn.Module):
    def __init__(self, args, src_emb_file=None, tgt_emb_file=None):
        super().__init__()
        self.src_vocab_size = args.src_vocab_size
        self.tgt_vocab_size = args.tgt_vocab_size
        self.vec_size = args.vec_size
        self.encoder_hidden_size = args.encoder_hidden_size
        self.decoder_hidden_size = args.decoder_hidden_size
        self.dropout_rate = args.dropout_rate
        self.start_ids = args.start_ids
        self.tgt_max_len = args.tgt_max_len
        self.teacher_forcing = getattr(args, "teacher_forcing", None)
        self.src_embedding = nn.Embedding(num_embeddings=self.src_vocab_size, embedding_dim=self.vec_size)
        self.tgt_embedding = nn.Embedding(num_embeddings=self.tgt_vocab_size, embedding_dim=self.vec_size)
        if src_emb_file is not None and os.path.exists(src_emb_file):
            with open(src_emb_file, "r") as emb_f:
                emb = np.array(json.load(emb_f), dtype=np.float32)
            self.src_embedding.weight.data.copy_(torch.from_numpy(emb))
            self.src_embedding.weight.requires_grad = True
        if tgt_emb_file is not None and os.path.exists(tgt_emb_file):
            with open(tgt_emb_file, "r") as emb_f:
                emb = np.array(json.load(emb_f), dtype=np.float32)
            self.tgt_embedding.weight.data.copy_(torch.from_numpy(emb))
            self.tgt_embedding.weight.requires_grad = True
        self.encoder = Encoder(self.src_embedding, self.vec_size, self.encoder_hidden_size, self.decoder_hidden_size,
                               self.dropout_rate)
        self.decoder = Decoder(self.tgt_embedding, self.tgt_vocab_size, self.vec_size, self.encoder_hidden_size,
                               self.decoder_hidden_size, self.dropout_rate)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")

    def forward(self, input_ids, labels=None):
        encoder_outputs, s = self.encoder(input_ids)
        device = input_ids.device
        if labels is None:
            logits = torch.zeros(input_ids.shape[0], self.tgt_max_len, self.tgt_vocab_size).to(device)
            predict_ids = torch.zeros(input_ids.shape[0], self.tgt_max_len).to(device)
            predict_ids[:, 0] = self.start_ids
            decoder_input_ids = torch.ones(input_ids.shape[0], dtype=torch.long).to(device) * self.start_ids
        else:
            logits = torch.zeros(input_ids.shape[0], labels.shape[1], self.tgt_vocab_size).to(device)
            predict_ids = torch.zeros(input_ids.shape[0], labels.shape[1]).to(device)
            decoder_input_ids = labels[:, 0]
        for t in range(1, logits.shape[1]):
            decoder_outputs, s = self.decoder(decoder_input_ids, s, encoder_outputs)
            arg_label = decoder_outputs.argmax(1)
            predict_ids[:, t] = arg_label
            logits[:, t, :] = decoder_outputs
            if labels is None:
                decoder_input_ids = arg_label
            else:
                teacher_force = random.random() < self.teacher_forcing
                decoder_input_ids = labels[:, t] if teacher_force else arg_label
        outputs = (predict_ids, logits,)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            num = labels.ne(0).long().sum().item()
            loss = loss / num
            outputs = (loss,) + outputs
        return outputs
