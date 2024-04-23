import torch
import torch.nn as nn
import os
import json
import numpy as np


class HighwayLayer(nn.Module):
    def __init__(self, input_size):
        super(HighwayLayer, self).__init__()
        self.normal_layer = nn.Linear(input_size, input_size)
        self.gate_layer = nn.Linear(input_size, input_size)

    def forward(self, input_emb):
        normal_layer_output = nn.functional.relu(self.normal_layer(input_emb))
        gate_layer_output = nn.functional.sigmoid(self.gate_layer(input_emb))
        output = torch.add(torch.mul(normal_layer_output, gate_layer_output),
                           torch.mul(input_emb, 1 - gate_layer_output))
        return output


class WordEmbeddingLayer(nn.Module):
    def __init__(self, emb_file, word_vocab_size, word_vec_size):
        super(WordEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=word_vec_size)
        if emb_file is not None and os.path.exists(emb_file):
            with open(emb_file, "r") as emb_f:
                emb = np.array(json.load(emb_f), dtype=np.float32)
            self.embedding.weight.data.copy_(torch.from_numpy(emb))
            self.embedding.weight.requires_grad = True

    def forward(self, input_ids):
        input_emb = self.embedding(input_ids)
        return input_emb


class CharEmbeddingLayer(nn.Module):
    def __init__(self, char_vocab_size, char_vec_size, dropout_rate):
        super(CharEmbeddingLayer, self).__init__()
        self.kernel_size = 5
        self.char_channel = char_vec_size
        self.char_vec_size = char_vec_size
        self.embed = nn.Embedding(num_embeddings=char_vocab_size, embedding_dim=self.char_vec_size)
        self.conv = nn.Conv2d(1, self.char_channel, (self.char_vec_size, self.kernel_size))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        batch_size = input_ids.shape[0]
        input_emb = self.dropout(self.embed(input_ids))
        input_emb = input_emb.view(-1, self.char_vec_size, input_emb.size(2)).unsqueeze(1)
        input_emb = self.conv(input_emb).squeeze()
        input_emb = nn.functional.max_pool1d(input_emb, input_emb.size(2)).squeeze()
        input_emb = input_emb.view(batch_size, -1, self.char_channel)
        return input_emb


class EmbeddingLayer(nn.Module):
    def __init__(self, emb_file, word_vocab_size, word_vec_size, char_vocab_size, char_vec_size, dropout_rate):
        super(EmbeddingLayer, self).__init__()
        self.char_embed = CharEmbeddingLayer(char_vocab_size, char_vec_size, dropout_rate)
        self.word_embed = WordEmbeddingLayer(emb_file, word_vocab_size, word_vec_size)
        self.highway1 = HighwayLayer(char_vec_size + word_vec_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.highway2 = HighwayLayer(char_vec_size + word_vec_size)

    def forward(self, input_ids_char, input_ids_word):
        input_char_emb = self.char_embed(input_ids_char)
        input_word_emb = self.word_embed(input_ids_word)
        input_emb = torch.cat((input_char_emb, input_word_emb), dim=-1)
        input_emb = self.highway1(input_emb)
        input_emb = self.dropout(input_emb)
        input_emb = self.highway2(input_emb)
        return input_emb


class ContextEmbeddingLayer(nn.Module):
    def __init__(self, vec_size, hidden_size, dropout_rate):
        super(ContextEmbeddingLayer, self).__init__()
        self.bilstm = nn.LSTM(input_size=vec_size,
                              hidden_size=hidden_size,
                              dropout=dropout_rate,
                              batch_first=True,
                              bidirectional=True)

    def forward(self, input_emb):
        lstm_output, _ = self.bilstm(input_emb)
        return lstm_output


class AttentionFlowLayer(nn.Module):
    def __init__(self, vec_size):
        super(AttentionFlowLayer, self).__init__()
        self.alpha = nn.Linear(3 * vec_size, 1, bias=False)

    def forward(self, context_emb, query_emb):
        shape = (context_emb.shape[0], context_emb.shape[1], query_emb.shape[1], context_emb.shape[2])
        context_extended = context_emb.unsqueeze(2).expand(shape)
        query_extended = query_emb.unsqueeze(1).expand(shape)
        cated = torch.cat((context_extended, query_extended, torch.mul(context_extended, query_extended)),
                          dim=-1)
        S = self.alpha(cated).view(shape[0], shape[1], shape[2])
        S_softmax_row = nn.functional.softmax(S, dim=2)
        S_max_col, _ = torch.max(S, dim=2)
        b_t = nn.functional.softmax(S_max_col, dim=1).unsqueeze(1)
        U_attention = torch.bmm(S_softmax_row, query_emb)
        h = torch.bmm(b_t, context_emb).squeeze()
        H_attention = h.unsqueeze(1).expand(shape[0], shape[1], shape[3])
        G = torch.cat((context_emb, U_attention, context_emb.mul(U_attention), context_emb.mul(H_attention)), dim=-1)
        return G


class ModelLayer(nn.Module):
    def __init__(self, vec_size, hidden_size, dropout_rate):
        super(ModelLayer, self).__init__()
        self.bilstm = nn.LSTM(input_size=vec_size,
                              hidden_size=hidden_size,
                              dropout=dropout_rate,
                              num_layers=2,
                              batch_first=True,
                              bidirectional=True)

    def forward(self, input_emb):
        lstm_output, _ = self.bilstm(input_emb)
        return lstm_output


class OutputLayer(nn.Module):
    def __init__(self, vec_size, hidden_size, dropout_rate):
        super(OutputLayer, self).__init__()
        self.start_w = nn.Linear(10 * vec_size, 1)
        self.bilstm = nn.LSTM(input_size=vec_size * 2,
                              hidden_size=hidden_size,
                              dropout=dropout_rate,
                              batch_first=True,
                              bidirectional=True)
        self.end_w = nn.Linear(10 * vec_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, G, M):
        GM = torch.cat((G, M), dim=-1)
        M2, _ = self.bilstm(M)
        GM2 = torch.cat((G, M2), dim=-1)
        p_start = self.start_w(GM).squeeze()
        p_end = self.end_w(GM2).squeeze()
        return p_start, p_end


class BiDAFModel(nn.Module):
    def __init__(self, args, emb_file=None):
        super(BiDAFModel, self).__init__()
        self.emb_layer = EmbeddingLayer(emb_file, args.word_vocab_size, args.word_vec_size, args.char_vocab_size,
                                        args.char_vec_size, args.dropout_rate)
        self.comtext_emb_layer = ContextEmbeddingLayer(args.char_vec_size + args.word_vec_size, args.hidden_size,
                                                       args.dropout_rate)
        self.attention_flow_layer = AttentionFlowLayer(args.hidden_size * 2)
        self.model_layer = ModelLayer(args.hidden_size * 8, args.hidden_size, args.dropout_rate)
        self.output_layer = OutputLayer(args.hidden_size, args.hidden_size, args.dropout_rate)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, context_input_ids, query_input_ids, context_char_input_ids, query_char_input_ids,
                start_labels=None, end_labels=None):
        context_emb = self.emb_layer(context_char_input_ids, context_input_ids)
        query_emb = self.emb_layer(query_char_input_ids, query_input_ids)
        context_emb = self.comtext_emb_layer(context_emb)
        query_emb = self.comtext_emb_layer(query_emb)
        G = self.attention_flow_layer(context_emb, query_emb)
        M = self.model_layer(G)
        start_logits, end_logits = self.output_layer(G, M)
        start_score = nn.functional.softmax(start_logits, dim=-1)
        end_score = nn.functional.softmax(end_logits, dim=-1)
        outputs = (start_score, end_score,)
        if start_labels is not None and end_labels is not None:
            start_loss = self.loss_fct(start_logits, start_labels)
            end_loss = self.loss_fct(end_logits, end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs
