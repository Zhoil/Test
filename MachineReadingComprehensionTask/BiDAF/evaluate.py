from collections import Counter
import string
import re
import sys


def normalize_answer(s):  # 对输入的字符串进行归一化处理

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)  # 将字符串中的冠词（a、an、the）替换为空格

    def white_space_fix(text):
        return ' '.join(text.split())  # 将字符串中的多个空格替换为单个空格

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)  # 获取所有的标点符号字符，并将字符串中的标点符号字符删除

    def lower(text):
        return text.lower()  # 将字符串转换为小写

    return white_space_fix(remove_articles(remove_punc(lower(s))))  # 返回


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    # 将预测结果和基准答案进行归一化处理，并使用split方法将其分割成词列表
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    # 创建prediction_tokens和ground_truth_tokens的计数器，并使用&操作符计算两个计数器的交集，即共同出现的词及其频次
    num_same = sum(common.values())  # 计算交集的频次总和，并将结果赋值给变量num_same
    if num_same == 0:
        return 0  # 表示预测结果和基准答案没有共同的词，直接返回F1分数为0
    precision = 1.0 * num_same / len(prediction_tokens)  # 计算预测结果的精确率
    recall = 1.0 * num_same / len(ground_truth_tokens)  # 计算预测结果的召回率
    f1 = (2 * precision * recall) / (precision + recall)  # 根据精确率和召回率计算F1分数
    return f1  # 返回结果


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))  # 用于计算预测结果和基准答案的精确匹配得分


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []  # 初始化一个空列表scores_for_ground_truths，用于存储每个基准答案与预测结果的得分
    for ground_truth in ground_truths:  # 循环遍历
        score = metric_fn(prediction, ground_truth)  # 计算预测结果与当前基准答案之间的得分
        scores_for_ground_truths.append(score)  # 将其添加到scores_for_ground_truths列表中
    return max(scores_for_ground_truths)  # 找到列表中的最大值，作为结果返回


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0  # 初始化了f1、exact_match和total三个变量，用于计算F1分数、精确匹配得分和总问题数
    for article in dataset:  # 遍历数据集中的每个文章
        for paragraph in article['paragraphs']:  # 遍历每个文章中的段落
            for qa in paragraph['qas']:  # 遍历段落中的每个问题
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue  # 如果问题的id在predictions字典中不存在，则打印一条错误信息，并将精确匹配得分设置为0
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]  # 如果问题的id存在于predictions字典中，则从问题的答案列表中提取基准答案的文本，并获取对应的预测结果
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
                # 分别计算预测结果和基准答案之间的精确匹配得分和F1分数，并使用metric_max_over_ground_truths方法找到最大值，并将其添加到exact_match和f1变量中

    exact_match = 100.0 * exact_match / total  # 计算精确匹配得分和F1分数的百分比
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}  # 将精确匹配得分和F1分数作为字典返回
