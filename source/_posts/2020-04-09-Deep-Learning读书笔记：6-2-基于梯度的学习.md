---
title: Deep Learning读书笔记：6.2 基于梯度的学习
urlname: deep-learning-chap-6-2-gradient-based-learning
mathjax: true
toc: true
date: 2020-04-09 23:14:28
updated: 2020-04-09 23:14:28
tags: [Deep Learning, Machine Learning]
categories: 读书笔记
---

设计和训练神经网络需要指定代价函数和输出单元。

<!--more-->

## 6.2.1 代价函数

大多数情况下，参数模型定义了分布$p(\boldsymbol{y} | \boldsymbol{x}; \boldsymbol{\theta})$，我们直接使用最大似然原理，即使用训练数据和模型预测之间的交叉熵作为代价函数。

有时我们不预测$\boldsymbol{y}$的完整概率分布，而仅预测在给定$\boldsymbol{x}$下$\boldsymbol{y}$的某种统计量。

完整的代价函数通常还要结合正则项。通常使用权重衰减方法。

### 6.2.1.1 使用最大似然学习条件分布

代价函数通常是负的对数似然，它和训练数据与模型分布之间的交叉熵等价，表示为

{% raw %}
$$
J(\boldsymbol{\theta})=-\mathbb{E}_{\mathbf{x}, \mathbf{y} \sim \hat{p}_{\mathrm{data}}} \log p_{\mathrm{model}}(\boldsymbol{y} | \boldsymbol{x})
$$
{% endraw %}

这种方法的优点是减轻了为每个模型设计代价函数的负担。

代价函数的梯度必须足够大，负对数似然的对数函数消除了某些输出单元的指数效果，可以避免输出单元饱和，梯度变小的问题。

交叉熵代价函数通常没有最小值，需要正则化技术进行修正。

### 6.2.1.2 学习条件统计量

有时我们并不是想学习一个完整的概率分布$p(\boldsymbol{y} | \boldsymbol{x}; \boldsymbol{\theta})$，而仅仅是想学习在给定$\boldsymbol{x}$时$\boldsymbol{y}$的某个条件统计量。从这个角度来看，可以把代价函数看成是一个**泛函**（functional），即函数到实数的映射。我们可以设计一个代价泛函，使得它在我们想要的某些特殊函数处取得最小值。对函数求解优化问题需要**变分法**（caculus of variations）。

使用变分法可以导出以下两个结果。

求解优化问题

{% raw %}
$$f^{*}=\underset{f}{\arg \min } \mathbb{E}_{\mathbf{x}, \mathbf{y} \sim p_{\mathrm{data}}}\|\boldsymbol{y}-f(\boldsymbol{x})\|^{2}$$
{% endraw %}

得到

{% raw %}
$$f^{*}(\boldsymbol{x})=\mathbb{E}_{\mathbf{y} \sim p_{\mathrm{data}}(\boldsymbol{y} | \boldsymbol{x})}[\boldsymbol{y}]$$
{% endraw %}

即最小化均方误差代价函数将得到一个函数，它可以用来对每个$\boldsymbol{x}$预测出$\boldsymbol{y}$的均值。

求解

{% raw %}
$$f^{*}=\underset{f}{\arg \min } \mathbb{E}_{\mathbf{x}, \mathbf{y} \sim p_{\text {data }}}\|\boldsymbol{y}-f(\boldsymbol{x})\|_{1}$$
{% endraw %}

将得到一个可以对每个$\boldsymbol{x}$预测$\boldsymbol{y}$取值的中位数的函数。这个代价函数通常被称为**平均绝对误差**（mean absolute error）。

但是，均方误差和平均绝对误差在使用基于梯度的优化方法时效果并不好，所以交叉熵代价函数比它们更受欢迎。

## 6.2.2 输出单元

代价函数的选择与输出单元的选择紧密相关。选择如何输出决定了交叉熵函数的形式。

本节中，假设前馈网络提供了一组定义为$\boldsymbol{h} = f(\boldsymbol{x}; \boldsymbol{\theta})$的隐藏特征。输出层的作用是对这些特征进行额外变换。

### 6.2.2.1 用于高斯输出分布的线性单元

给定特征$\boldsymbol{h}$，线性输出单元层产生一个向量$\hat{y}=W^{\top} h+b$。

线性输出层经常用来产生条件高斯分布的均值：

$$p(\boldsymbol{y} | \boldsymbol{x})=\mathcal{N}(\boldsymbol{y} ; \hat{\boldsymbol{y}}, \boldsymbol{I})$$

最大化其对数似然此时等价于最小化均方误差。

由于线性模型不会饱和，所以它们易于采用基于梯度的优化算法。

### 6.2.2.2 用于Bernoulli输出分布的sigmoid单元

许多任务需要预测二值型变量$y$的值，此时最大似然的方法是定义$y$在$\boldsymbol{x}$条件下的Bernoulli分布。

使用sigmoid单元结合最大似然保证模型给出错误答案时，总能有一个较大的梯度。sigmoid输出单元定义为

$$\hat{y}=\sigma\left(\boldsymbol{w}^{\top} \boldsymbol{h}+b\right)$$

可以认为sigmoid输出单元具有两个部分：首先使用一个线性层来计算$z = \boldsymbol{w}^{\top} \boldsymbol{h}+b$，然后使用sigmoid激活函数将$z$转化为概率。

sigmoid激活函数参数化Bernoulli分布时，只会在模型已经得到正确答案时饱和，对于极度不正确的情况则完全不会收缩梯度，这使得基于梯度的模型可以很快地改正错误的$z$。

### 6.2.2.3 用于Multinoulli输出分布的softmax单元

当我们想要表示一个具有$n$个可能取值的离散型随机变量的分布时，我们可以使用softmax函数，它可以看做是sigmoid函数的扩展。softmax函数最常用作分类器的输出，表示$n$个不同类上的概率分布。

softmax函数的形式为

{% raw %}
$$\boldsymbol{z}=W^{\top} \boldsymbol{h}+\boldsymbol{b}$$

$$\operatorname{softmax}(\boldsymbol{z})_{i}=\frac{\exp \left(z_{i}\right)}{\sum_{j} \exp \left(z_{j}\right)}$$
{% endraw %}

使用最大化对数似然训练softmax输出目标值$y$时，使用指数函数工作得非常好。

### 6.2.2.4 其他的输出类型

// TODO
