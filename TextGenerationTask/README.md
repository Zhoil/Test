<h1 align="center">
  文本生成
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
    <li><a href="#GRUATT"> ➤ GRUATT</a>
      <ul>
        <li><a href="#gruatt-intro"> ➤ 介绍</a></li>
        <li><a href="#gruatt-train"> ➤ 训练</a>
        <ul>
          <li><a href="#gruatt-train-preprocess">数据预处理</a></li>
          <li><a href="#gruatt-train-model">模型训练</a></li>
          <li><a href="#gruatt-train-predict">预测</a></li>
          <li><a href="#gruatt-train-caution">注意事项</a></li>
        </ul>
        </li>
        <li><a href="#gruatt-result"> ➤ 实验结果</a></li>
      </ul>
    </li>
    <li>
      <a href="#mbartT"> ➤ MBART</a>
      <ul>
        <li><a href="#mbart-intro"> ➤ 介绍</a></li>
        <li><a href="#mbart-train"> ➤ 训练</a>
        <ul>
          <li><a href="#mbart-train-preprocess">数据预处理</a></li>
          <li><a href="#mbart-train-model">模型训练</a></li>
          <li><a href="#mbart-train-predict">预测</a></li>
          <li><a href="#mbart-train-caution">注意事项</a></li>
        </ul>
        </li>
        <li><a href="#mbart-result"> ➤ 实验结果</a></li>
      </ul>
    </li>
    <!--<li><a href="#experiments">Experiments</a></li>-->
  </ul>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# GRUATT

<h2 id="gruatt-intro">📖 介绍</h2>

基于GRU+Attention机制的Seq2Seq结构，用于进行机器翻译任务。

<h2 id="gruatt-train">⚙️ 训练</h2>

🔔 注意：运行代码时，请将目录切到GRUATT目录下运行

<h3 id="gruatt-train-preprocess">2.1 数据预处理</h3>

首先进行IWSLT14EN-DE的下载和预处理：

```shell
sh prepare_iwslt14.sh
```

然后进行数据预处理：

```python
python3 data_helper.py
```

🔔 注意

- 数据为IWSLT14 EN-DE数据
- 数据下载后的处理过程中，通过mosedecoder以及subword-nmt工具包进行了一些筛选和替换处理，最后得到了原始文本以及经过应用BPE分词后的数据
- 当前模型使用的应用BPE分词后的数据
- `data_helper.py`处理过程中，仅为了构建词表、标签和词向量，并没有对原始文本进行其他处理，同时为了便于学习，默认只取20k条数据进行训练。
- 如有需要，可以按照命名标准，使用外部Embedding进行初始化

<h3 id="gruatt-train-model">2.2 模型训练</h3>

```python
python3 main.py
or
python3 main.py --device "0"
```

🔔注意:

- 如果修改其他参数，请在main.py文件中修改，或在执行命令上增加参数即可。
- 注意保持args中和语言相关的参数和对应文件的后缀保持一致（例如:`--src-lang en`和`train.en`保持一致）

<h3 id="gruatt-train-predict">2.3 预测</h3>

```python
python3 predict.py
or
python3 predict.py --device "0" --model_path "output_dir/checkpoint-epoch2-bs-8-lr-0.001"
```

🔔注意:

- 需要额外安装`sacrebleu`:`pip install sacrebleu`来计算bleu值

- 模型预测时，需要加载训练好得模型，控制参数model_path。

<h3 id="gruatt-train-caution">2.4 注意事项</h3>

📺 上述三个步骤呈pipeline执行，需要保证和`src_lang`和`tgt_lang`参数是一致的

🚚 如果调节参数，保存了不同的模型，需要更新`model_path`参数

✌️ 如果需要更换数据，请参照`prepare_iwslt14.sh`进行格式转换

<h2 id="gruatt-result">📝 实验结果</h2>

⭐️ 使用的计算指标为BLEU

🐳 以下结果为默认参数训练得到

| Dataset | BLEU |
| ------- | -------- |
| IWSLT14 EN-DE   | 5.76   |

到这里意味着运行成功✌️！该结果不是最佳结果，希望大家能调出更好的结果～🎉

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# BERT

<h2 id="mbart-intro">📖 介绍</h2>

论文：https://arxiv.org/pdf/2001.08210.pdf

<h2 id="mbart-train">⚙️ 训练</h2>

🔔 注意：运行代码时，请将目录切到MBART目录下运行

<h3 id="mbart-train-preprocess">2.1 数据预处理</h3>

首先进行IWSLT14EN-DE的下载和预处理：

```shell
sh prepare_iwslt14.sh
```

然后进行数据预处理：

```python
python3 data_helper.py
```

🔔 注意:

- 数据为IWSLT14 EN-DE数据
- 数据下载后的处理过程中，通过mosedecoder以及subword-nmt工具包进行了一些筛选和替换处理，最后得到了原始文本以及经过应用BPE分词后的数据
- 这里使用的原始文本数据
- `data_helper.py`处理过程中，仅为了构建词表、标签和词向量，并没有对原始文本进行其他处理，同时为了便于学习，默认只取20k条数据进行训练。


<h3 id="mbart-train-model">2.2 模型训练</h3>

```
python3 main.py
or
python3 main.py --device "0"
``` 

🔔注意:

- 如果修改其他参数，请在main.py文件中修改，或在执行命令上增加参数即可。
- 模型训练，需要加载预训练MBART模型[相关文件](https://huggingface.co/facebook/mbart-large-50/tree/main)，并放到对应目录`data/mbart_large_50`下。
- 对于可以设置`--pre-train-model`的参数为`facebook/mbart-large-50`直接通过接口下载模型参数.
- 在开始训练前需要安装`sentencepiece`:`pip install sentencepiece`
- 注意保持args中和语言相关的参数和对应文件的后缀保持一致（例如：`--src-lang en`和`train.en`保持一致）
- 更改MBART相关语言参数（例如：`mbart_src_lang_code`）请到[相关网站](https://huggingface.co/facebook/mbart-large-50)进行查阅进行查阅

<h3 id="mbart-train-predict">2.3 预测</h3>

```
python3 predict.py
or
python3 predict.py --device "0" --model_path "output_dir/checkpoint-epoch0-bs-4-lr-2e-05"
``` 

🔔注意:

- 模型预测时，需要加载训练好得模型，控制参数model_path。

<h3 id="mbart-train-caution">2.4 注意事项</h3>

📺 上述三个步骤呈pipeline执行，需要保证task参数是一致的

🚚 如果调节参数，保存了不同的模型，需要更新`model_path`参数

⭐️ 如果需要更换数据，请参照`prepare_iwslt14.sh`进行格式转换


<h2 id="mbart-result">📝 实验结果</h2>

✌️ 使用的计算指标为BLEU

🐳 以下结果为默认参数训练得到

| Dataset | BLEU |
| ------- | -------- |
| IWSLT14 EN-DE   | 11.27   |

到这里意味着运行成功✌️！该结果不是最佳结果，希望大家能调出更好的结果～🎉