# <center><font face='楷体'><font size='12'>机器学习/深度学习的相关笔记</font></font></center>

### <font face='宋体'>前言:</font>

&emsp;&emsp;<font face='黑体'>*欢迎来到此处,这里是我边学习边~~整理~~的有关机械学习/深度学习的相关笔记。先前我对这方面的知识不是很了解,笔记整理必然有不妥之处,请见谅并斧正。*</font>

<center><img src="https://tse1-mm.cn.bing.net/th/id/OIP-C.CkyhRmyZgQ9QaMSc7n7WawHaHa?w=215&h=215&c=7&r=0&o=5&dpr=1.3&pid=1.7"></center>

-----

目录：
### [深度自然语言处理](#1)
##### [1.Word Vectors(词向量)](#11)
##### [2.Neural Classifiers(神经分类器)](#12)
##### [3.神经网络和反向传播](#13)
##### [4.Dependency Parsing](#14)
##### [5.语言模型(LM)和循环神经网络(RNNs)](#15)
##### [6.LSTM](#16)
##### [7.机器翻译、Seq2Seq和注意力机制(attention)](#17)
##### [8.NLP中的问答系统](#18)
##### [9.NLP中的卷积神经网络](#19)
##### [10.NLP子词模型](#110)
##### [11.基于上下文的表征与NLP预训练模型](#11)
### [NLP与深度学习的未来](#2)

-----------------

# <font size='6'><center><div id="1">深度自然语言处理(NLP)</div></center></font>

&emsp;&emsp;<font size='4'> ***人类的语言有什么特别之处？人类语言是一个专门用来表达意义的系统,语言文字是上层抽象表征,NLP与计算机视觉或任何其他机器学习任务都有很大的不同***
&emsp;&emsp;**~~语言是混乱的,自然处理的目的是更好地理解语言中的意思及其影响。~~**
</font>

        近年来在深度学习中比较有效的方式是基于上下文的词汇表征。它的核心想法是:一个单词的意思是由经常出现在它附近的单词给出的。

$●当一个单词 \omega 出现在文本中时，它的上下文是出现在其附近的一组单词(在一个固定大小的窗口中)$

$●基于海量数据，使用 \omega 的许多上下文来构建 \omega 的表示$

**NLP模型算法就是研究如何让计算机读懂人类语言，即将人的自然语言转换为计算机可以阅读的指令。**

-------

## <font size='5'><div id="11">1.Word Vectors(词向量)</font>

<center><img src="https://pic.imgdb.cn/item/650ce421c458853aef14331a.png"></center>


<font size='4'>***Word Vectors是利用向量来表示单词, 并可以从中筛选相似度高的不同单词以及其他衍生的比较和选择方法。***

***使用词向量编码单词, N 维空间足够我们编码语言的所有语义,每一维度都会编码一些我们使用语言传递的信息。***

***处理词向量之前,必要的公式应该得到演示,如下：***
**$$P( O | C )=\frac{exp(u_o^\tau v_c)}{\sum_{w\in Vocab}exp(u_w^\tau v_c)}$$**

###### 注：上述为在word2vec中,条件概率分布公式,是通过取向量点积并应用来给出的<font color='blue'>softmax</font>最大功能。

        Word2Vec是一个迭代模型,该模型能够根据文本进行迭代学习,并最终能够对给定上下文的单词的概率对词向量进行编码呈现,而不是计算和存储一些大型数据集(可能是数十亿个句子)的全局信息.

-------

## <font size='5'><span id="12">2.Neural Classifiers(神经分类器)</span></font>


对于分类问题,我们有训练**数据集**:它由一些**样本**组成
**$$\{x_i,y_i\}^N_{i=1}$$**
●$x_i$**输入**,例如单词(索引或是向量),句子,文档等等(维度为 **$d$**)
●$y_i$是我们尝试预测的**标签**($C$ 个类别中的一个),例如:
&emsp;&emsp;⊙类别:感情,命名实体,购买/售出的决定。
&emsp;&emsp;⊙其他单词。
&emsp;&emsp;⊙多词序列。

训练数据 **$\{x_i,y_i\}^N_{i=1}$**
### &emsp;●**softmax分类器:** 
&emsp;在softmax分类器中最常用到交叉熵损失，也是负对数概率形态。
&emsp;对于每个训练样本(x,y),我们的目标是最大化正确类$y$的概率,或者我们可以**最小化该类的负对数概率**
**$$ -logp(y|x)=-log(\frac{exp(f_y)}{\sum^C_{c=1}exp(f_c)}) $$**
使用对数概率将我们的目标函数转换为求和形态,这更容易在推导和应用中使用。

    注:交叉熵的损失理解
●交叉熵的概念来源于信息论,衡量两个分布之间的差异
●令真实概率分布为$p$,我们计算的模型概率分布为$q$
●交叉熵为
**$$ H(p,q)=-\sum_{c=1}^Cp(c)logq(c) $$**
●假设标准答案的概率分布是,在正确类上为1,在其他类别上为0:
**$$ p=[0,\cdots,0,1,0,\cdots,0] $$**
●因为 **$p$** 是独热向量,所以唯一剩下的项是真实的负对数概率
### &emsp;●神经网络分类器
&emsp;&emsp;单独使用线性分类器$softmax(≈logisti回归)$并不强大
<center><img src="https://pic.imgdb.cn/item/650ce5cbc458853aef148f1d.png"></center>
&emsp;&emsp;如上图所示,$softmax$得到的是线性决策边界
&emsp;&emsp;&emsp;⊙对于复杂问题来说，它的表达能力是有限的
&emsp;&emsp;&emsp;⊙有一些分错的点，需要更强的非线性表达能力来区分
**我们需要非线性决策边界来支持更高级的分类需要**

-------

## <font size='5'><div id="13">3.神经网络和反向传播</font>


&emsp;●**实体命名识别(NER)**
&emsp;&emsp;跟踪文档中提到的特定实体
&emsp;&emsp;对于问题回答,答案通常是命名实体
&emsp;&emsp;许多需要的信息实际上是命名实体之间的关联
&emsp;&emsp;同样的技术可以扩展到其他 slot-filling 槽填充分类

&emsp;&emsp;**通常后面是命名实体链接/规范化到知识库**

&emsp;●**句子中的命名实体识别**
&emsp;&emsp;我们通过在上下文中对单词进行分类,然后将实体提取为单词子序列来预测实体。

&emsp;●**NER难点**

    很难计算出实体的边界
    很难知道某物是否是一个实体
    很难知道未知/新奇实体的类别
    实体类是模糊的,依赖于上下文


##### 权重矩阵的梯度导数:
**$$ s=u^Th $$** **$$ h=f(z) $$** **$$ z=Wx+b $$**
#### **计算$\frac{\delta s}{\delta W} $**
#### **使用链式法则**
#### **$$ \frac{\delta s}{\delta W}=\frac{\delta s}{\delta h} \frac{\delta h}{\delta z} \frac{\delta z}{\delta W} $$**

##### 反向传播的梯度求导:
<center><img src="https://pic.imgdb.cn/item/650ce5cbc458853aef148f29.png" alt="image-2.png"></center>

<center><img src="https://pic.imgdb.cn/item/650ce5cbc458853aef148f36.png" alt="image-3.png"></center>

### Question:应该使用可用的“预训练”词向量吗？
#### answer:
    几乎总是「应该用」
    他们接受了大量的数据训练,所以他们会知道训练数据中没有的单词,也会知道更多关于训练数据中的单词
    拥有上亿的数据语料吗？那可以随机初始化开始训练

##### 计算图与反向传播:
我们把神经网络方程表示成一个图
<center><img src="https://pic.imgdb.cn/item/650ce5cbc458853aef148f4e.png"></center>


--------

## <font size='5'><div id="14">4.Dependency Parsing</font>

#### 成分与依赖:
&emsp;&emsp;句子是使用逐步嵌套的单元构建的
&emsp;&emsp;短语结构将单词组织成嵌套的成分
<center><img src="https://pic.imgdb.cn/item/650ce6a8c458853aef14ad73.png" alt="image-6.png"></center>

&emsp;&emsp;**起步单元:** 单词被赋予一个类别
&emsp;&emsp;**单词**组合成不同类型的词语
&emsp;&emsp;**短语**可以递归地组合成更大地短语

&emsp;&emsp;单词的排列可以组成许多意义的句子
&emsp;&emsp;但是同时,单个句子所表达的含义也会**有所不同**
&emsp;&emsp;面对复杂结构的句子,我们需要考虑指数级的可能结构,这个序列被称为**卡特兰数**
**$$ C_n=\frac{(2n)!}{(n+1)!n!} $$**


#### 依赖语法与树库:
**Dependency Structure有两种表现形式**
⊙直接在句子上标出依存关系箭头及语法关系
⊙将其做成树状机构(Dependency Tree Graph)
<center><img src="https://pic.imgdb.cn/item/650ce6a8c458853aef14ad81.png" alt="image-7.png"></center>

    箭头通常标记(type)为语法关系的名称(主题、介词对象、apposition等)
    箭头连接头部(head)(调速器，上级，regent)和一个依赖(修饰词，下级，下属)
        A->的事情
    通常，依赖关系形成一棵树(单头，无环，连接图)

-------

## <font size='5'><div id="15">5.语言模型(LM)和循环神经网络(RNNs)</font>

#### 语言模型:
<center><img src="https://pic.imgdb.cn/item/650ce6a8c458853aef14ad8d.png" alt="image-8.png"></center>

**语言建模**的任务是预测下一个单词是什么
更正式的说法是:给定一个单词序列$ x^{(1)},x^{(2)},\dots,x^{(t)} $,计算下一个单词$ x^{(t+1)} $的概率分布:
**$$ P(x^{(t+1)}|x^{(t)},\dots,x^{(1)}) $$**
&emsp;&emsp;其中,**$x^{(t+1)}$** 可以是词表中的任意单词 **$ V=\{w_1,\dots,w_{|V|} \} $**
&emsp;&emsp;这样做的系统被称为 Language Model 语言模型
<center><img src="https://pic.imgdb.cn/item/650ce6a9c458853aef14ada4.png" alt="image-9.png"></center>
<center><img src="https://pic.imgdb.cn/item/650ce6a9c458853aef14adb5.png" alt="image-10.png"></center>

#### RNNs:
<img src="https://pic.imgdb.cn/item/650ce733c458853aef14f388.png" alt="image-11.png">

**核心想法:** 重复使用相同的权重矩阵 **$W$**
<img src="https://pic.imgdb.cn/item/650ce734c458853aef14f3a6.png" alt="image-12.png">

●**RNN优点:**
&emsp;&emsp;可以处理**任意长度**的输入
&emsp;&emsp;**模型大小不会**随着输入的增加而**增加**
&emsp;&emsp;在每个时间步上应用相同的权重,因此在处理输入时具有**对称性**
●**RNN缺点:**
&emsp;&emsp;循环串行计算速度慢
&emsp;&emsp;在实践中,很难从许多步骤前返回信息

##### 训练模型:
获取一个**较大的文本语料库**,该语料库是一个单词序列
输入RNN-LN;计算 **每个步骤$t$** 的输出分布

    即预测到目前为止给定的每个单词的概率分布

步骤$t$上的**损失函数**为预测概率分布 **$ \hat{y}^{(t)} $** 与真实下一个单词 **$ y^{(t)}(x^{(t+1)}的独热向量) $** 之间的**交叉熵**
**$$ J^{(t)}(\theta)=CE(y^{(t)},\hat{y}^{(t)})=-\sum_{w\in V}y^{(t)}_w log \hat{y}^{(t)}_w = -log \hat{y}^{(t)}_{x_{t+1}} $$**
将其平均,得到整个训练集的**总体损失**
**$$ J(\theta)=\frac{1}{T} \sum_{t=1}^{T} J^{(t)}(\theta) = \frac{1}{T} \sum_{t=1}^{T} -log \hat{y}^{(t)}_{x_{t+1}} $$**

-------

## <font size='5'><div id="16">6.LSTM</font>

#### 长短时记忆(LSTM):

<img src="https://pic.imgdb.cn/item/650ce734c458853aef14f3bf.png" alt="image-13.png">

Hochreiter和Schmidhuber在1997年提出了一种RNN，用于解决梯度消失问题。
在第 **$t$** 步,有一个**隐藏状态** **$h^{(t)}$** 和一个**单元状态** **$c^{(t)}$**
&emsp;&emsp;都是长度为n的向量
&emsp;&emsp;单元储存长期信息
&emsp;&emsp;LSTM可以从单元中**擦除、写入**和**读取信息**
信息被 擦除/写入/读取 的选择由三个对应的门控制
&emsp;&emsp;门也是长度为 **$n$** 的向量
&emsp;&emsp;在每个时间步长上,门的每个元素可以**打开**(1)、**关闭**(0)或介于两者之间
&emsp;&emsp;**门是动态的:** 它们的值是基于当前上下文计算的
<img src="https://pic.imgdb.cn/item/650ce734c458853aef14f3e8.png" alt="image-14.png">

--------

## <font size='5'><div id="17">7.机器翻译、Seq2Seq和注意力机制(attention)</font>


这里我们重点关注神经网路机器翻译

    神经机器翻译(NMT)是利用单个神经网络进行机器翻译的一种方法
    神经网络架构称为 sequence-to-sequence (又名seq2seq)，它包含两个RNNs

**sequence-to-sequence** 模型是条件语言模型的一个例子
<img src="https://pic.imgdb.cn/item/650ce734c458853aef14f40e.png" alt="image-15.png">

#### 注意力机制:
Sequence-to-sequence：瓶颈问题
<img src="https://pic.imgdb.cn/item/650ce795c458853aef1534f2.png" alt="image-16.png">

**注意力**为瓶颈问题提供了一个解决方案

**核心理念:** 在解码器的每一步,使用与**编码器的直接连接**来专注于源序列的**特点部分**

首先我们将通过图表展示,然后我们将用方程
<img src="https://pic.imgdb.cn/item/650ce796c458853aef153500.png" alt="image-17.png">
<img src="https://pic.imgdb.cn/item/650ce796c458853aef15350d.png" alt="image-18.png">

**注意力最后的性能:**
&emsp;&emsp;注意力显著提高了**NMT性能**

        这是非常有用的，让解码器专注于某些部分的源语句

&emsp;&emsp;注意力解决**瓶颈问题**

        注意力允许解码器直接查看源语句；绕过瓶颈

&emsp;&emsp;注意力**帮助消失梯度问题**

        提供了通往遥远状态的捷径

&emsp;&emsp;注意力**提供了一些可解释性**

        通过检查注意力的分布，我们可以看到解码器在关注什么
        我们可以免费得到(软)对齐
        网络只是自主学习了对齐

## <font style="background:red">注意力是一种普遍的深度学习技巧</font>

-------

## <font size='5'><div id="18">8.NLP中的问答系统</font>

#### 我们今天要讨论的不是基于结构化储存的问答。
**~~而是在一段或一份文件中找到答案~~**

    这个问题通常被称为阅读理解
    这就是我们今天要关注的

<img src="https://pic.imgdb.cn/item/650ce796c458853aef153519.png" alt="image-19.png">

<img src="https://pic.imgdb.cn/item/650ce796c458853aef153533.png" alt="image-20.png">

#### 复杂的系统，但他们在<font color="brown">事实</font>问题上做得相当好
**非常复杂的多模块多组件的系统:**

    首先对问题进行解析，使用手写的语义规范化规则，将其转化为更好的语义形式
    在通过问题类型分类器，找出问题在寻找的语义类型
    信息检索系统找到可能包含答案的段落，排序后进行选择
    NER识别候选实体再进行判断

**这样的QA系统在特定领域很有效：Factoid Question Answering 针对实体的问答**


#### BiDAF:
<img src="https://pic.imgdb.cn/item/650ce7e5c458853aef15438f.png" alt="image-21.png">
<img src="https://pic.imgdb.cn/item/650ce7e6c458853aef15439a.png" alt="image-22.png">

多年来,BiDAF architecture有许多变体和改进,但其核心思想是 **the Attention Flow layer**
**思路:** attention 应该双向流动——从上下文到问题,从问题到上下文
<img src="https://pic.imgdb.cn/item/650ce7e6c458853aef1543b0.png" alt="image-23.png">

------

## <font size='5'><div id="19">9.NLP中的卷积神经网络</font>

#### 从RNN到CNN：
循环神经网络不能捕获没有前缀上下文的短语
经常在最终向量中捕获的信息太多来自于最后的一些词汇内容
例如: $softmax$ 通常只在最后一步计算

**那么，什么是卷积？**
&emsp;●一维离散卷积一般为: **$ (f*g)[n]= \sum\limits_{m=-M}^{M} f[n-m]g[m] $**
&emsp;●卷积通常地用于从图像中提取特征

        模型位置不变的识别

●二维示例:

        黄色和红色数字显示过滤器 (=内核) 权重
        绿色显示输入
        粉色显示输出

<img src="https://pic.imgdb.cn/item/650ce7e6c458853aef1543be.png" alt="image-24.png">
<img src="https://pic.imgdb.cn/item/650ce7e6c458853aef1543dd.png" alt="image-25.png">

对比RNN与CNN:
<img src="https://pic.imgdb.cn/item/650ce80ac458853aef1548dd.png" alt="image-26.png">

-------

## <font size='5'><div id="110">10.NLP子词模型</font>

<img src="https://pic.imgdb.cn/item/650ce80bc458853aef1548e8.png" alt="image-27.png">


------

## <font size='5'><div id="111">11.基于上下文的表征与NLP预训练模型</font>

#### 预训练的词向量:
POS和NER两种表征体系
11个词窗，100个隐层神经元，在12w词上训练7周
<img src="https://pic.imgdb.cn/item/650ce80bc458853aef1548f5.png" alt="image-28.png">
<img src="https://pic.imgdb.cn/item/650ce80bc458853aef15490c.png" alt="image-29.png">

简单且常见的解决方案:

&emsp;**训练时:** 词汇表$ \{ word occurring,say, ≥5\ times \} \ \cup \{ <UNK> \} $
&emsp;&emsp;将所有罕见的词 (数据集中出现次数小于5) 都映射为$<UNK>$,为其训练一个词向量
&emsp;**运行时:** 使用$<UNK>$代替词汇表之外的词OOV

●**问题:**
&emsp;没有办法区分不同 UNK words,无论是身份还是意义
<img src="https://pic.imgdb.cn/item/650ce80bc458853aef154924.png" alt="image-30.png">

#### transformer:
<img src="https://pic.imgdb.cn/item/650ce828c458853aef154d86.png" alt="image-31.png">
<img src="https://pic.imgdb.cn/item/650ce828c458853aef154d99.png" alt="image-32.png">

#### BERT
**BERT** 使用 mask 的方式进行整个上下文的预测,使用了双向的上下文信息
<img src="https://pic.imgdb.cn/item/650ce828c458853aef154da6.png" alt="image-33.png">

--------

## 尚未完结--------

## <font size='5'><div id="2">NLP与深度学习的未来</font>

#### 为何近年来深度学习的如此成功？
&emsp;&emsp;**扩展能力(模型和数据大小)** 是深度学习近些年来成功的很大一部分原因

    同时,过去受到计算资源和数据资源的规模限制,深度学习一直处于尚未成熟发展的地位

无标签和无监督学习的不断发展和进步
###### 注:**神经网络在NLP深度学习中的广泛发展已经成为了这以领域不可或缺的一部分**

计算机视觉可视化的推动与多任务处理方式的改进

**低资源支撑的场景:**

    不需要很多计算能力的模型(这对移动设备尤为重要)
    低资源语言
    低数据环境(ML中的元学习越来受欢迎)

### NLP正逐渐对社会产生巨大影响力









**<p align="right"><font color="blue">Used by Han</font></p>**
**<p align="right"><font color="brown">2023</font></p>**
[Learn.md](https://github.com/Zhoil/Test/files/15455023/Learn.md)
[Learn.md](https://github.com/Zhoil/Test/files/15455023/Learn.md)
率公式应该得到演示，如下：***
**$$P( O | C )=\frac{exp(u_o^\tau v_c)}{\sum_{w\in Vocab}exp(u_w^\tau v_c)}$$**

###### 注：上述为在word2vec中，条件概率分布公式，是通过取向量点积并应用来给出的<font color='blue'>softmax</font>最大功能。
