import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import numpy as np


class TextCNN(nn.Module):
    def __init__(self, args, emb_file=None):
        super(TextCNN, self).__init__()
        self.vocab_size = args.vocab_size  # 初始化词汇表大小
        self.vec_size = args.vec_size  # 初始化词向量维度
        self.max_len = args.max_len  # 初始化最大文本长度
        self.filter_num = args.filter_num
        self.in_channels = args.in_channels
        self.dropout_rate = args.dropout_rate
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.vec_size)  # 创建一个 Embedding 层，用于将词索引映射为词向量
        if emb_file is not None and os.path.exists(emb_file):
            with open(emb_file, "r") as emb_f:
                emb = np.array(json.load(emb_f), dtype=np.float32)
            self.embedding.weight.data.copy_(torch.from_numpy(emb))
            self.embedding.weight.requires_grad = True
        # 创建三个卷积层，分别针对不同大小的卷积核进行卷积操作
        self.conv_1 = self.conv_layer(in_channels=args.in_channels, filter_num=self.filter_num, kernel=args.kernels[0],
                                      vec_size=self.vec_size, max_len=self.max_len)
        self.conv_2 = self.conv_layer(in_channels=args.in_channels, filter_num=self.filter_num, kernel=args.kernels[1],
                                      vec_size=self.vec_size, max_len=self.max_len)
        self.conv_3 = self.conv_layer(in_channels=args.in_channels, filter_num=self.filter_num, kernel=args.kernels[2],
                                      vec_size=self.vec_size, max_len=self.max_len)
        # 创建一个线性层，将卷积层的输出进行全连接，并将输出的维度设置为 args.filter_num * 3，即三个卷积层的输出拼接起来
        self.linear = nn.Linear(in_features=args.filter_num * 3, out_features=args.label_number)
        self.dropout = nn.Dropout(p=args.dropout_rate)  # 创建一个 Dropout 层，用于在训练过程中进行随机失活
        self.loss_fct = nn.CrossEntropyLoss()  # 创建一个交叉熵损失函数，用于计算模型的损失

    def conv_layer(self, in_channels, filter_num, kernel, vec_size, max_len):
        conv = nn.Sequential(  # 创建一个序列化的模型 conv
            nn.Conv2d(in_channels=in_channels, out_channels=filter_num, kernel_size=(kernel, vec_size)),  # 创建一个二维卷积层
            nn.ReLU(),  # 使用 ReLU 激活函数对卷积层的输出进行非线性变换
            nn.MaxPool2d(kernel_size=(max_len - kernel + 1, 1)),  # 使用最大池化操作 nn.MaxPool2d 对卷积层输出的特征图进行池化
        )
        return conv  # 返回这个卷积层模型

    def forward(self, input_ids, labels=None):  # 前向传播函数
        input_emb = self.embedding(input_ids)  # 将输入文本的词索引通过嵌入层 self.embedding 转换为词向量表示
        input_emb = self.dropout(input_emb).unsqueeze(1)  # 对词向量进行随机失活操作 self.dropout ，将随机失活后的词向量增加一个维度
        conv1_out = self.conv_1(input_emb).squeeze()  # 将增加维度后的词向量通过卷积层卷积层的输出
        conv2_out = self.conv_2(input_emb).squeeze()  # 将增加维度后的词向量通过卷积层卷积层的输出
        conv3_out = self.conv_3(input_emb).squeeze()  # 将增加维度后的词向量通过卷积层卷积层的输出
        conv_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=-1)  # 将三个卷积层的输出在最后一个维度上进行拼接
        conv_out = self.dropout(conv_out)  # 对拼接后的输出进行随机失活操作
        logits = self.linear(conv_out)  # 将随机失活后的输出通过线性层 self.linear 得到分类的 logits
        score = nn.functional.softmax(logits)  # 使用 softmax 函数对 logits 进行归一化，得到分类的概率 score
        outputs = (score,)  # 将概率 score 存入元组 outputs 中
        if labels is not None:
            loss = self.loss_fct(logits, labels)  # 计算模型预测值与真实标签之间的交叉熵损失
            outputs = (loss,) + outputs  # 将损失存入元组 outputs 中
        return outputs  # 返回元组 outputs
