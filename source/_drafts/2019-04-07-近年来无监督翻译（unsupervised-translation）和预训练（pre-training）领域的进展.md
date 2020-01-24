---
title: 近年来NLP领域半监督学习、无监督翻译和预训练（pre-training）领域的进展
urlname: nlp-semi-supervised-learning-unsupervised-translation-and-pre-training
toc: true
mathjax: true
date: 2019-04-07 16:29:55
updated: 2019-04-07 16:29:55
tags: [NLP]
categories: NLP
---

我预计这篇文章要坑，不过我只想综述一下它们之间的关系而已。

本文将谈到以下几篇论文：

<!--more-->

* [Improving Neural Machine Translation Models with Monolingual Data (Sennrich et al. 2015)](http://arxiv.org/abs/1511.06709)
* [Neural Machine Translation of Rare Words with Subword Units (Sennrich et al. 2015)](http://arxiv.org/abs/1508.07909)
* [Dual Learning for Machine Translation (He et al. 2016)](http://papers.nips.cc/paper/6469-dual-learning-for-machine-translation)
* [Learning Distributed Representations of Sentences from Unlabelled Data (Hill et al. 2016)](http://arxiv.org/abs/1602.03483)
* [Unsupervised Pretraining for Sequence to Sequence Learning (Ramachandran et al. 2016)](http://arxiv.org/abs/1611.02683)
* [Enriching Word Vectors with Subword Information (Bojanowski et al. 2017)](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00051)
* [Word Translation Without Parallel Data (Conneau et al. 2017)](http://arxiv.org/abs/1710.04087)
* [Learning bilingual word embeddings with (almost) no bilingual data (Artetxe et al. 2017)](https://www.aclweb.org/anthology/P17-1042)
* [Unsupervised Machine Translation Using Monolingual Corpora Only (Lample et al. 2017)](http://arxiv.org/abs/1711.00043)
* [Unsupervised Neural Machine Translation (Artetxe et al. 2017)](http://arxiv.org/abs/1710.11041)
* [Deep contextualized word representations (Peters et al. 2018)](https://arxiv.org/abs/1802.05365)
* [GPT: Improving Language Understanding by Generative Pre-Training (Radford et al. 2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al. 2018)](http://arxiv.org/abs/1810.04805)
* [Language Models are Unsupervised Multitask Learners (Radford et al. 2018)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* [Phrase-Based & Neural Unsupervised Machine Translation (Lample et al. 2018)](http://arxiv.org/abs/1804.07755)
* [Xnli: Evaluating cross-lingual sentence representations (Conneau et al. 2018)](https://arxiv.org/abs/1809.05053)
* [Cross-lingual Language Model Pretraining (Lample et al. 2019)](http://arxiv.org/abs/1901.07291)
* [Multi-Task Deep Neural Networks for Natural Language Understanding (Liu et al. 2019)](http://arxiv.org/abs/1901.11504)

## [Improving Neural Machine Translation Models with Monolingual Data (Sennrich et al. 2015)](http://arxiv.org/abs/1511.06709)

这篇文章提出了用back-translation进行数据增广的方法，这一方法后来被应用在很多无监督翻译任务中。

不过我还没看这篇文章。不明白细节

- [ ]: 读文章

## [Neural Machine Translation of Rare Words with Subword Units (Sennrich et al. 2015)](http://arxiv.org/abs/1508.07909)

这篇文章提出了用BPE对语料进行预处理的方法，它目前已经成为了主流预处理方法，并衍生出了（

- [ ]: BPE都衍生出了什么？

BPE本身和神经网络没什么关系，它只是把一般的分词扩展到了分字母上，然后把词切成了subword而已。

## [Dual Learning for Machine Translation (He et al. 2016)](http://papers.nips.cc/paper/6469-dual-learning-for-machine-translation)

这篇文章好像首次提出了无监督翻译（是无监督吗？？）中对偶学习的概念。

我还没看。

- [x]: 读文章

## [Learning Distributed Representations of Sentences from Unlabelled Data (Hill et al. 2016)](http://arxiv.org/abs/1602.03483)

这篇文章比较了各种无监督训练生成distributed sentence vector的方法，并提出了Sequential Denoising Autoencoder（SDAE）和fastText两种新的训练方法。结果我忘了。

- [ ]: 所以新的训练方法和旧的有何差异？

所谓distributed sentence vector和一般所说的预训练word vector是非常类似的概念，只不过是句子的vector。当然，句子vector可以直接用词来组成，但这样不一定能捕捉到句子内部的语义结构，所以作者希望能够通过其他的训练objective，得到更好的sentence vector。

其中值得注意的就是这两种新的训练方法。Sequential Denoising Autoencoder（SDAE）是Bengio提出的Denoising Autoencoders的修改版（[Extracting and composing robust features with denoising autoencoders (Vincent et al. 2008)](http://portal.acm.org/citation.cfm?doid=1390156.1390294)）。原版的DAE是将训练数据的一部分随机置零然后再重建，这里针对NLP任务的特点，改成了这样：

定义noise function$N(S | p_o, p_x)$，其中$p_o$和$p_x$是超参数，分别表示删除和交换的概率。对于句子$S$中的每一个词$\omega_i$，以$p_o$概率删除$\omega_i$；然后对于每一对互不重合的bigram$\omega_i \omega_{i+1}$，以$p_x$概率交换$\omega_i \omega_{i+1}$。然后就可以在噪音数据上进行训练了。（但是他没给训练目标的公式，我不记得这个过程要不要被back-propagate了 // TODO）

然而我也没好好看最后表示取的是什么，是autoencoder的中间高层representation吗？// TODO

FastSent的idea比较平凡，像是词袋模型和“句袋模型”的结合体：context是前一句+后一句，句子的embedding $s_i$即其中的word embedding的和：

$$s_i = \sum_{\omega \in S_i} u_{\omega}$$

训练时，每个例子的loss就是句子embedding和每个context word的embedding的softmax的和：

$$\sum_{\omega \in S_{i-1} \cup S_{i+1}} \phi(s_i, v_{\omega})$$

我认为这个FastSent的idea和[Enriching Word Vectors with Subword Information (Bojanowski et al. 2017)](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00051)提出的fastText非常类似，只不过一个是在句子和词上训练，一个是在词和字母上训练而已。然而他们并没有互相引用。

到目前为止，无监督句子embedding可能不是研究的重点（word embedding会被用来做模型初始化，sentence embedding不知道拿来干啥……），但是这篇文章中提到的SDAE的方法是比较好的。

- [ ]: SDAE用在哪些工作上了？

以及，这是Kyunghyun Cho的研究工作。

## [Unsupervised Pretraining for Sequence to Sequence Learning (Ramachandran et al. 2016)](http://arxiv.org/abs/1611.02683)

把这篇文章列举出来是因为历史原因。

## [Enriching Word Vectors with Subword Information (Bojanowski et al. 2017)](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00051)

这大概是一个BPT + skipgram + word embedding的混合体。

[Word Translation Without Parallel Data (Conneau et al. 2017)](http://arxiv.org/abs/1710.04087)就引用了这篇文章作为基础，用它得到的无监督单语词向量作为对齐双语词向量的基础。

## [Word Translation Without Parallel Data (Conneau et al. 2017)](http://arxiv.org/abs/1710.04087)

这大概是无监督词典。

## [Learning bilingual word embeddings with (almost) no bilingual data (Artetxe et al. 2017)](https://www.aclweb.org/anthology/P17-1042)

这大概是弱监督词典。

（Artetxe和Lample最近简直在对着干，也许再过几天我就能看到他们那边的预训练成果了……）

## [Unsupervised Machine Translation Using Monolingual Corpora Only (Lample et al. 2017)](http://arxiv.org/abs/1711.00043)

我发现这篇文章也做了一个很好的综述，里面列举了很多无监督的paper，其中有一些我应该阅读一下；而且它也列出了一个理论框架，是我应该去学习的；甚至应该新开一篇文章。

（以及我发现投ICLR的文章一般综述都写得比较好哎）

相关工作包括：

* 半监督：
  * 用target侧单语数据进行数据增益（[Improving Neural Machine Translation Models with Monolingual Data (Sennrich et al. 2015)](http://arxiv.org/abs/1511.06709)）
  * 在target侧增加语言模型
  * 使用单语数据附加autoencoding objective（[Dual Learning for Machine Translation (He et al. 2016)](http://papers.nips.cc/paper/6469-dual-learning-for-machine-translation)）
* 无监督：
  * 利用相近语言对
  * 利用other modalities（这是啥？）

- [ ]: 研究other modalities是什么

---

而作者的想法是这样的：为了能够训练出翻译模型，我们只需保证两种语言共享latent space，然后训练模型，使得对于一种语言的有噪声的句子，它不仅可以重建句子（SDAE），还可以重建它在另一种语言中正确的被翻译版本（back-translation）。为了强制共享latent representation，作者使用了对抗训练方法。模型的初始化使用的是他们之前的工作中提出的无监督词典模型（[Word Translation Without Parallel Data (Conneau et al. 2017)](http://arxiv.org/abs/1710.04087)）。

（这些内容在他们之后的文章中被大大细化了。[Phrase-Based & Neural Unsupervised Machine Translation (Lample et al. 2018)](http://arxiv.org/abs/1804.07755)）

用denoising autoencoder重建句子的训练方法和[Learning Distributed Representations of Sentences from Unlabelled Data (Hill et al. 2016)](http://arxiv.org/abs/1602.03483)类似，但此处作者给出了（写得非常繁复的公式）：

![式(1)](denoising-objective-form.png)

这个式子本质上就是说，$x$是句子，记$C(x)$为加噪声之后的句子，$\hat{x}$为模型重建出的句子，最小化重建结果和$x$的差异。噪声模型$C(x)$包括drop word和*slightly* shuffle两个步骤，和[Learning Distributed Representations of Sentences from Unlabelled Data (Hill et al. 2016)](http://arxiv.org/abs/1602.03483)相比略有差别。

- [ ]: 把这篇文章写完。。。

## [Unsupervised Neural Machine Translation (Artetxe et al. 2017)](http://arxiv.org/abs/1710.11041)

以及这篇文章也是Kyunghyun Cho的工作，也许和那篇[Learning Distributed Representations of Sentences from Unlabelled Data (Hill et al. 2016)](http://arxiv.org/abs/1602.03483)会更相像一些吧。

## [Deep contextualized word representations (Peters et al. 2018)](https://arxiv.org/abs/1802.05365)

这篇文章就是ELMo。它做的是一种“高级word embedding”：学到的东西是bi-LSTM的内部状态。这实际上就是一种预训练了，所以之后它会被和OpenGPT以及BERT比较。

## [GPT: Improving Language Understanding by Generative Pre-Training (Radford et al. 2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

GPT是最近第一个提出预训练的模型吗？也许是，也许不是，我不记得了。

但不管怎么说，它论文写得不错。相关工作包括三种类型：

* NLP半监督学习：
  * 用无标注数据计算word/phrase-level统计值，当做feature
  * 用无标注数据训练word embedding
  * 用无标注数据训练phrase/sentence-level embedding（此处引用了[Unsupervised Machine Translation Using Monolingual Corpora Only (Lample et al. 2017)](http://arxiv.org/abs/1711.00043)作为例子）
* 无监督预训练：
  * 首先用LM objective进行训练，然后有监督调参
  * 主要问题是它们用的都是LSTM
* 附加训练objective

这使得我现在想修改一下文章结构，并且加上一篇。

## [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al. 2018)](http://arxiv.org/abs/1810.04805)

这就是大名鼎鼎的BERT了。

## [Language Models are Unsupervised Multitask Learners (Radford et al. 2018)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

GPT-2之前引起了一些争议。

## [Phrase-Based & Neural Unsupervised Machine Translation (Lample et al. 2018)](http://arxiv.org/abs/1804.07755)

这篇文章实际上也是G. Lample的工作，可以看作是[Unsupervised Machine Translation Using Monolingual Corpora Only (Lample et al. 2017)](http://arxiv.org/abs/1711.00043)的延续。

更好的无监督翻译。

## [Xnli: Evaluating cross-lingual sentence representations (Conneau et al. 2018)](https://arxiv.org/abs/1809.05053)

### 问题

目前大部分相关文章都是monolingual的。

### 相关工作

* XLU（corss-lingual language understanding）：系统主要在一种语言上进行训练，然后在其他语言上进行测试
* NLI（natural language inference）：判断两个句子之间的关系为entailment、contradiction还是neutral

### 实验方法

创建了一个XLU测试集，将MNLI的开发集和测试集翻译成15种语言，称为XNLI，并且实现了很多baseline。

测试集和开发集共有7500个句对，翻译后变成了112500对。

用XNLI测试了一些在训练时使用平行数据的sentence encoder。当然，XNLI也可以测试general-purpose的sentence encoder。

### 实验结果

## [Cross-lingual Language Model Pretraining (Lample et al. 2019)](http://arxiv.org/abs/1901.07291)

甚至更好的无监督翻译。

还是Lample的工作。

## [Multi-Task Deep Neural Networks for Natural Language Understanding (Liu et al. 2019)](http://arxiv.org/abs/1901.11504)

代码：[namisan/mt-dnn](https://github.com/namisan/mt-dnn)

是之前的[Representation Learning Using Multi-Task Deep Neural Networks for Semantic Classification and Information Retrieval](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/mltdnn.pdf)的延续。

### 问题

### 相关工作

### 方法

在之前工作的基础上加上了一个BERT。

### 实验结果

获得了很多SOTA。