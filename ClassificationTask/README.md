<h1 align="center">
  文本分类
</h1>
<p align="center">
  <!-- version -->
  <img src='https://img.shields.io/badge/PyTorch-v1.8.0-red' />
  <!-- transformers  -->
  <img src='https://img.shields.io/badge/Transformers-v4.4.1-orange' />
  <a href="https://img.shields.io/badge/version-v0.1.0-blue">
      <img alt="version" src="https://img.shields.io/badge/version-v0.1.0-blue?color=FF8000?color=009922" />
    </a>
  <a >
       <img alt="Status-building" src="https://img.shields.io/badge/Status-building-blue" />
  	</a>
</p>
<h2 id="table-of-contents"> 📖 目录</h2>

<details open="open">
  <summary>目录内容</summary>
  <ul>
    <li><a href="#TextCNN"> ➤ TextCNN</a>
      <ul>
        <li><a href="#textcnn-intro"> ➤ 介绍</a></li>
        <li><a href="#textcnn-train"> ➤ 训练</a>
        <ul>
          <li><a href="#textcnn-train-preprocess">数据预处理</a></li>
          <li><a href="#textcnn-train-model">模型训练</a></li>
          <li><a href="#textcnn-train-predict">预测</a></li>
          <li><a href="#textcnn-train-caution">注意事项</a></li>
        </ul>
        </li>
        <li><a href="#textcnn-result"> ➤ 实验结果</a></li>
      </ul>
    </li>
    <li>
      <a href="#Bert"> ➤ Bert</a>
      <ul>
        <li><a href="#bert-intro"> ➤ 介绍</a></li>
        <li><a href="#bert-train"> ➤ 训练</a>
        <ul>
          <li><a href="#bert-train-preprocess">数据预处理</a></li>
          <li><a href="#bert-train-model">模型训练</a></li>
          <li><a href="#bert-train-predict">预测</a></li>
          <li><a href="#bert-train-caution">注意事项</a></li>
        </ul>
        </li>
        <li><a href="#bert-result"> ➤ 实验结果</a></li>
      </ul>
    </li>
    <!--<li><a href="#experiments">Experiments</a></li>-->
  </ul>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# TextCNN

<h2 id="textcnn-intro">📖 介绍</h2>

论文：https://arxiv.org/pdf/1408.5882.pdf

<h2 id="textcnn-train">⚙️ 训练</h2>

🔔 注意：运行代码时，请将目录切到TextCNN目录下运行

<h3 id="textcnn-train-preprocess">2.1 数据预处理</h3>

首先先将SNIPS数据集进行数据格式转换：

```
python3 preprocess.py
```

然后进行数据预处理：

```python
python3 data_helper.py --task SNIPS
or
python3 data_helper.py --task SST2
```

🔔 注意

- 该数据预处理中，仅为了构建词表、标签和词向量，并没有对原始文本进行其他处理，如果需要去除停用词等，请自行添加代码。
- 运行前，查看是否存在"data/glove.6B.100d.txt"文件，如果不存在，则模型得embedding将随机初始化。"glove.6B.100d.txt"文件[下载地址](https://github.com/liucongg/NLPCodeCourse/blob/main/PyTorch_Code/ClassificationTask)。
- Github中提供的数据为SNIPS和SST2数据。

<h3 id="textcnn-train-model">2.2 模型训练</h3>

```python
python3 main.py --task SNIPS
or
python3 main.py --task SST2
```

🔔注意:

- 如果修改其他参数，请在main.py文件中修改，或在执行命令上增加参数即可。

<h3 id="textcnn-train-predict">2.3 预测</h3>

```python
python3 predict.py --task SNIPS --model_path "output_dir/SNIPS/checkpoint-epoch2-bs-50-lr-0.0005"
or
python3 predict.py --task SST2 --model_path "output_dir/SST2/checkpoint-epoch4-bs-50-lr-0.0005"
```

🔔注意:

- 模型预测时，需要加载训练好得模型，控制参数model_path。

<h3 id="textcnn-train-caution">2.4 注意事项</h3>

📺 上述三个步骤呈pipeline执行，需要保证`task`参数是一致的

🚚 如果调节参数，保存了不同的模型，需要更新`model_path`参数

✌️ 如果需要更换数据，请参照`preprocess.py`进行格式转换

<h2 id="textcnn-result">📝 实验结果</h2>

⭐️ 使用的计算指标为accuracy

🐳 以下结果为默认参数训练得到

| Dataset | Accuracy |
| ------- | -------- |
| SINPS   | 0.9643   |
| SST2    | 0.8320   |

到这里意味着运行成功✌️！该结果不是最佳结果，希望大家能调出更好的结果～🎉

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# BERT

<h2 id="bert-intro">📖 介绍</h2>

论文：https://arxiv.org/pdf/1810.04805.pdf

<h2 id="bert-train">⚙️ 训练</h2>

🔔 注意：运行代码时，请将目录切到BERT目录下运行

<h3 id="bert-train-preprocess">2.1 数据预处理</h3>

首先先将SNIPS数据集进行数据格式转换：

```
python3 preprocess.py
```
然后进行数据预处理：
```python
python3 data_helper.py --task SNIPS
or
python3 data_helper.py --task SST2
```

🔔 注意:

- 该数据预处理中，仅为了构建标签，并没有对原始文本进行其他处理，如果需要去除停用词等，请自行添加代码。
- Github中提供的数据为**SST2**数据和**SNIPS**数据。

<h3 id="bert-train-model">2.2 模型训练</h3>

```python
python3 main.py --task SNIPS
or
python3 main.py --task SST2
```

🔔注意:

- 如果修改其他参数，请在main.py文件中修改，或在执行命令上增加参数即可。
- 模型训练，需要加载预训练BERT模型[下载链接](https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin)，并放到对应目录`data/bert_base`下。
- 对于可以设置`--pre-train-model`的参数为`bert-base-uncased`直接通过接口下载模型参数.

<h3 id="bert-train-predict">2.3 预测</h3>

```python
python3 predict.py --task SNIPS --model_path "output_dir/SNIPS/checkpoint-epoch2-bs-8-lr-2e-05"
or
python3 predict.py --task SST2 --model_path "output_dir/SST2/checkpoint-epoch4-bs-8-lr-2e-05"
```

🔔注意:

- 模型预测时，需要加载训练好得模型，控制参数model_path。

<h3 id="bert-train-caution">2.4 注意事项</h3>

📺 上述三个步骤呈pipeline执行，需要保证task参数是一致的

🚚 如果调节参数，保存了不同的模型，需要更新`model_path`参数

⭐️ 如果需要更换数据，请参照preprocess.py进行格式转换


<h2 id="bert-result">📝 实验结果</h2>

✌️ 使用的计算指标为accuracy

🐳 以下结果为默认参数训练得到

| Dataset | Accuracy |
| ------- | -------- |
| SINPS   | 0.9714   |
| SST2    | 0.9204   |

到这里意味着运行成功✌️！该结果不是最佳结果，希望大家能调出更好的结果～🎉