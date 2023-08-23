# <center><font face='楷体'><font size='12'>机器学习/深度学习的相关笔记</font></font></center>

### <font face='宋体'>前言:</font>

&emsp;&emsp;<font face='黑体'>*欢迎来到此处，这里是我边学习边~~整理~~的有关机械学习/深度学习的相关笔记。先前我对这方面的知识不是很了解，笔记整理必然有不妥之处，请见谅并斧正。*</font>

<center><img src="https://tse1-mm.cn.bing.net/th/id/OIP-C.CkyhRmyZgQ9QaMSc7n7WawHaHa?w=215&h=215&c=7&r=0&o=5&dpr=1.3&pid=1.7"></center>

# <font size='5'>一.深度自然语言处理</font>
-----------------
&emsp;&emsp;<font size='4'> ***对人类来说, 自然语言的处理一般是在现实的基础上由人工加载而成。因此在机械深度学习中，如何将人类的各种自然语言翻译为计算机语言，交由电脑处理是重要的课题。***
&emsp;&emsp;**~~语言是混乱的，自然处理的目的是更好地理解语言中的意思及其影响。~~**
## <font size='4'>1.Word Vectors(词向量)</font>
&emsp;&emsp;<font size='4'> ***![Alt text](image.png)***
***Word Vectors是利用向量来表示单词, 并可以从中筛选相似度高的不同单词以及其他衍生的比较和选择方法。***

***处理词向量之前，必要的概率公式应该得到演示，如下：***
**$$P( O | C )=\frac{exp(u_o^\tau v_c)}{\sum_{w\in Vocab}exp(u_w^\tau v_c)}$$**

###### 注：上述为在word2vec中，条件概率分布公式，是通过取向量点积并应用来给出的<font color='blue'>softmax</font>最大功能。
