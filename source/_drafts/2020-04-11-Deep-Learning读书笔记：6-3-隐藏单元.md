---
title: Deep Learning读书笔记：6.3 隐藏单元
urlname: deep-learning-chap-6-3-hidden-units
mathjax: true
toc: true
date: 2020-04-11 16:45:54
updated: 2020-04-11 16:45:54
tags: [Deep Learning, Machine Learning]
categories: 读书笔记
---

隐藏单元的设计是一个活跃的研究领域。隐藏单元最好选择整流线性单元。这里将描述对于每种隐藏单元的一些基本直觉。

<!--more-->

虽然一些隐藏单元并不是在所有的输入点上都可微，但梯度下降在实践中仍然表现得足够好。部分原因是神经网络训练算法通常不会达到代价函数的局部最小值。

大多数隐藏单元都可以描述为接受输入向量$\boldsymbol{x}$，计算仿射变换$z=W^{\top} x+b$，然后应用一个逐元素的非线性函数$g(\boldsymbol{z})$。大多数隐藏单元的区别仅在于$g(\boldsymbol{z})$的形式。

## 6.3.1 整流线性单元及其扩展

整流线性单元使用激活函数{% raw %}$g(z)=\max \{0, z\}${% endraw %}。它们和线性单元非常类似，只要整流线性单元处于激活状态，它的梯度不仅大而且一致。

初始化仿射变换的参数时，可以将$\boldsymbol{b}$的所有元素设置成一个小的正值，比如0.1，这使得