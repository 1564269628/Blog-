---
title: Deep Learning读书笔记：6.1 实例：学习XOR
urlname: deep-learning-chap-6-1-example-learning-xor
mathjax: true
toc: true
date: 2020-04-09 17:20:43
updated: 2020-04-09 17:20:43
tags: [Deep Learning, Machine Learning]
categories: 读书笔记
---

首先介绍一个简单的前馈网络，它的功能是学习XOR函数。

<!--more-->

XOR函数是两个二进制值$x_1$和$x_2$之间的运算，当这两个值中恰好有一个为1时，XOR函数返回值是1，否则为0。

我们考察网络在这四个点上的表现：{% raw %}$\mathbb{X}=\left\{[0,0]^{\top},[0,1]^{\top},[1,0]^{\top},[1,1]^{\top}\right\}${% endraw %}

我们可以把这个问题当做是回归问题，并使用均方误差损失函数，这样，评估整个训练集上表现的MSE损失函数为

$$J(\boldsymbol{\theta})=\frac{1}{4} \sum_{x \in \mathbb{X}}\left(f^{*}(\boldsymbol{x})-f(\boldsymbol{x} ; \boldsymbol{\theta})\right)^{2}$$

下面选择模型$f(\boldsymbol{x}; \boldsymbol{\theta})$的形式。如果我们选择一个线性模型，会发现无法解决这个问题，因为XOR问题是线性不可分的。我们可以使用一个模型来学习一个不同的特征空间，在这个空间上线性模型能够表示这个解。引入一个有一个隐藏层的前馈神经网络：

{% raw %}
$$
\begin{aligned}
\boldsymbol{h} &= f^{(1)}(\boldsymbol{x} ; \boldsymbol{W}, \boldsymbol{c}) \\
y &= f^{(2)}(\boldsymbol{h} ; \boldsymbol{w}, b) \\
f(\boldsymbol{x} ; \boldsymbol{W}, \boldsymbol{c}, \boldsymbol{w}, b) &= f^{(2)}\left(f^{(1)}(\boldsymbol{x})\right)
\end{aligned}
$$
{% endraw %}

如果$f^{(1)}$是线性函数，则整个网络仍然是线性的，因此需要让它是非线性函数。定义$\boldsymbol{h}=g\left(\boldsymbol{W}^{\top} \boldsymbol{x}+\boldsymbol{c}\right)$，其中$g$是对每个元素分别起作用的函数，在现代神经网络中，默认使用$g(z)=\max \{0, z\}$定义的**整流线性单元**（rectified linear unit），又称ReLU；于是整个网络变为

{% raw %}
$$
f(\boldsymbol{x} ; \boldsymbol{W}, \boldsymbol{c}, \boldsymbol{w}, b)=\boldsymbol{w}^{\top} \max \left\{0, \boldsymbol{W}^{\top} \boldsymbol{x}+\boldsymbol{c}\right\}+b
$$
{% endraw %}

下面可以给出XOR问题的一个解。令

{% raw %}
$$
\begin{aligned}
\boldsymbol{W} &=\left[\begin{array}{cc}
1 & 1 \\
1 & 1
\end{array}\right] \\
\boldsymbol{c} &=\left[\begin{array}{c}
0 \\
-1
\end{array}\right] \\
\boldsymbol{w} &=\left[\begin{array}{c}
1 \\
-2
\end{array}\right] \\
b &= 0
\end{aligned}
$$
{% endraw %}

模型处理一批输入的方法是，将每个输入置于矩阵的每一行，得到输入矩阵

{% raw %}
$$
\boldsymbol{X} =\left[\begin{array}{cc}
0 & 0\\
0 & 1\\
1 & 0\\
1 & 1
\end{array}\right]
$$
{% endraw %}

神经网络的第一步是将第一层的权重矩阵乘以输入矩阵：

{% raw %}
$$
\boldsymbol{XW} =\left[\begin{array}{cc}
0 & 0\\
1 & 1\\
1 & 1\\
2 & 2
\end{array}\right]
$$
{% endraw %}

然后加上偏置向量$\boldsymbol{c}$（注意此处是广播式加法），得到

{% raw %}
$$
\left[\begin{array}{cc}
0 & -1\\
1 & 0\\
1 & 0\\
2 & 1
\end{array}\right]
$$
{% endraw %}

对上述结果使用整流线性变换，得到

{% raw %}
$$
\boldsymbol{h} = \left[\begin{array}{cc}
0 & 0\\
1 & 0\\
1 & 0\\
2 & 1
\end{array}\right]
$$
{% endraw %}

最后乘以权重向量$\boldsymbol{w}$，得到

{% raw %}
$$
\left[\begin{array}{c}
0\\
1\\
1\\
0
\end{array}\right]
$$
{% endraw %}

在这个例子中，我们直接猜出了解决方案；基于梯度的优化算法可以找到一些参数，使得产生的误差非常小，我们此处给出的解位于损失函数的全局最小值，因此梯度下降算法可以收敛到这一点。
