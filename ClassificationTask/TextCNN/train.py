import os
from data_set import collate_func
from torch.utils.data import DataLoader
import torch
import logging
import numpy as np
from tqdm import tqdm, trange

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)  # 配置日志记录的基本设置
logger = logging.getLogger(__name__)  # 获取日志对象并赋值给logger


def train(args, model, device, train_data, dev_data):
    # args 命令行参数的实例，用于传递训练的配置参数
    # model 待训练的模型
    # device 指定的设备，用于模型的计算
    # train_data 训练数据集
    # dev_data 验证数据集
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)  # 检查输出目录是否存在，如果不存在则创建
    tb_write = SummaryWriter()  # 记录训练过程的日志和指标
    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.train_batch_size,
                              collate_fn=collate_func,
                              shuffle=True)  # 按批次加载和处理训练数据
    model.to(device)  # 将模型移动到指定的设备上进行计算
    paras = model.parameters()  # 将模型移动到指定的设备上进行计算
    optimizer = torch.optim.Adam(paras, lr=args.learning_rate)  # 获取模型的参数和配置优化器
    model.zero_grad()  # 对模型的梯度进行清零
    max_acc = 0.

    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):  # 进行多个训练轮数的循环，每轮中遍历训练数据集的每个批次
        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        model.train()
        for step, batch in enumerate(iter_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss = model(input_ids, labels)[0]
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        eval_acc, eval_loss = evaluate(args, model, device, dev_data)
        tb_write.add_scalar("dev_acc", eval_acc, i_epoch)
        tb_write.add_scalar("dev_loss", eval_loss, i_epoch)
        logging.info(
            "epoch is {}, dev_acc is {}, dev_loss is {}".format(i_epoch, eval_acc, eval_loss))  # 打印当前轮次的验证精度和损失值
        if eval_acc > max_acc:  # 如果当前验证精度超过历史最佳精度，则保存模型的参数到指定目录中
            max_acc = eval_acc
            output_dir = os.path.join(args.output_dir,
                                      'checkpoint-epoch{}-bs-{}-lr-{}'.format(i_epoch, args.train_batch_size,
                                                                              args.learning_rate))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))
        torch.cuda.empty_cache()  # 清空 GPU 缓存
    logger.info('Train done')  # 训练结束后，使用 logger 记录日志信息，表示训练完成


def evaluate(args, model, device, dev_data):  # 对模型在验证集上的性能进行评估，计算预测准确率和损失值，并返回评估结果
    # args 命令行参数的实例，用于传递评估的配置参数
    # model 待评估的模型
    # device 指定的设备，用于模型的计算
    # dev_data 验证数据集
    dev_data_loader = DataLoader(dev_data, batch_size=args.dev_batch_size, collate_fn=collate_func)  # 创建验证数据的 DataLoader 对象 dev_data_loader，用于按批次加载和处理验证数据
    iter_bar = tqdm(dev_data_loader, desc="iter", disable=False)  # 创建进度条 iter_bar，用于显示评估进度
    y_true = []  # 初始化空列表 y_true 和 y_predict，用于存储真实标签和预测标签
    y_predict = []  # 初始化空列表 y_true 和 y_predict，用于存储真实标签和预测标签
    total_loss, total = 0.0, 0.0  # 初始化总损失值 total_loss 和样本数量 total
    for step, batch in enumerate(iter_bar):  # 遍历验证数据集的每个批次
        model.eval()  # 将模型切换到评估模式
        with torch.no_grad():  # 不计算梯度，进行前向传播获取模型的输出
            input_ids = batch["input_ids"].to(device)  # 移动输入数据和标签到指定设备
            labels = batch["labels"].to(device)  # 移动输入数据和标签到指定设备
            y_true.extend(labels.cpu().numpy().tolist())  # 将真实标签存储到 y_true 列表中
            outputs = model.forward(input_ids=input_ids, labels=labels)
            y_label = torch.argmax(outputs[1], dim=-1)  # 使用 argmax 函数获取预测标签，并将其存储到 y_predict 列表中
            y_predict.extend(y_label.cpu().numpy().tolist())
            loss = outputs[0].item()  # 计算当前批次的损失值
            total_loss += loss * len(batch["input_ids"])  # 计算当前批次的损失值，并累加到总损失值 total_loss 中
            total += len(batch["input_ids"])  # 累加样本数量 total
    y_true = np.array(y_true)  # 将 y_true 转换为 numpy 数组
    y_predict = np.array(y_predict)  # 将 y_predict 转换为 numpy 数组
    eval_acc = np.mean((y_true == y_predict))  # 计算预测准确率，即真实标签和预测标签相等的样本比例
    eval_loss = total_loss / total  # 计算平均损失值，即总损失值除以样本数量
    return eval_acc, eval_loss  # 返回评估结果
