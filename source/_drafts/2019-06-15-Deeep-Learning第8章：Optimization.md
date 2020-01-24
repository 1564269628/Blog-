---
title: Deeep Learning第8章：Optimization
urlname: Deeep Learning第8章：Optimization
toc: true
mathjax: true
date: 2019-06-15 02:13:43
updated: 2019-06-15 02:13:43
tags:
categories:
---

本章讨论的是以降低神经网络的代价函数$J(\mathbf{\theta})$为目标的优化。

概述：

* 作为机器学习任务训练算法的优化和单纯的优化有何不同
* 使得神经网络优化困难的原因
* 实用算法
* 先进算法
* 复合优化策略

## 学习和优化的区别

最大的区别是，机器学习通常不是直接优化对应的目标，而是通过降低$J(\mathbf{\theta})$来间接优化目标。

下式是在训练集上的代价函数。

$$J(\boldsymbol{\theta})=\mathbb{E}_{(\boldsymbol{x}, \mathrm{y}) \sim \hat{p}_{\mathrm{data}}} L(f(\boldsymbol{x} ; \boldsymbol{\theta}), y)$$

如果是在整个数据空间上计算代价，则上式就会变成

$$J^{*}(\boldsymbol{\theta})=\mathbb{E}_{(\boldsymbol{x}, \mathrm{y}) \sim p_{\mathrm{data}}} L(f(\boldsymbol{x} ; \boldsymbol{\theta}), y)$$

### 经验风险最小化

这种代价函数的定义相当于每次都在整个训练集上训练。

机器学习算法的目标通常是降低**泛化误差的期望值**，或称**风险**；但我们不知道数据的真实分布，否则最小化风险就变成了普通的优化问题；因此我们最小化训练集上的**误差的期望值**，或称**经验风险**（其实就是损失的平均值），并希望风险也随之大幅度下降（这概念可是绕了几个弯）。这种方法被称为**经验风险最小化**。

$$\text{经验风险} = \mathbb{E}_{\boldsymbol{x}, \mathrm{y} \sim \hat{p}_{\text { data }}(\boldsymbol{x}, y)}[L(f(\boldsymbol{x} ; \boldsymbol{\theta}), y)]=\frac{1}{m} \sum_{i=1}^{m} L\left(f\left(\boldsymbol{x}^{(i)} ; \boldsymbol{\theta}\right), y^{(i)}\right)$$

这种方法有两个问题：

* 容易过拟合
* 大多数优化算法基于梯度下降，但很多损失函数没有导数，如0-1 loss[^01]

[^01]: 0-1 loss指的是这样的一种分类loss：在测试集上分类错误几个，loss就是几。这样的loss显然是没有科学的导数的，因为在大部分时候导数都为0，只在几个点上导数为无穷大。[Why is the 0-1 indicator function a poor choice for loss function?](https://qr.ae/TWheA6)

## 替代损失函数和早停

首先可以对损失函数没有导数这一点进行改进：使用**替代损失函数**（surrogate loss function）。
