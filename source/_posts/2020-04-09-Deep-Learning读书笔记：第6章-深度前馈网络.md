---
title: Deep Learning读书笔记：第6章 深度前馈网络
urlname: deep-learning-chap-6-deep-feedforward-networks
toc: true
mathjax: true
date: 2020-04-09 16:51:09
updated: 2020-04-09 16:51:09
tags: [Deep Learning, Machine Learning]
categories: 读书笔记
---

**深度前馈网络**（deep feedforward network）：又称**前馈神经网络**（feedforward neural network）或**多层感知机**（multilayer perceptron，MLP），是典型的深度学习模型，其目标是近似某个函数$f^*$。这种模型被称为前向的，是因为在模型的输入和输出之间没有反馈（feedback）连接。

前馈神经网络被称为**网络**（network）是因为，它通常是不同函数复合在一起构成的，如$f(\boldsymbol{x})=f^{(3)}\left(f^{(2)}\left(f^{(1)}(\boldsymbol{x})\right)\right)$，其中$f^{(1)}$称为网络的**第一层**（first layer），$f^{(2)}$称为网络的**第二层**（second layer），以此类推。复合链的全长称为模型的**深度**（depth）。前馈网络的最后一层称为**输出层**（output layer），它的目标是产生一个接近标签的值；其它层所需的输出没有被直接给出，它们被称为**隐藏层**（hidden layer）。网络中的隐藏层的维数决定了模型的**宽度**（width）。

除了把层想象成向量到向量的单个函数，我们也可以把层想象成许多并行操作的**单元**（unit），每个单元表示一个向量到标量的函数。每个单元类似于一个神经元，它接收的输入来源于许多其他单元，并计算它自己的激活值。

下面考虑一种用线性模型解释前馈网络的方法。为了扩展线性模型来表示$\boldsymbol{x}$的非线性函数，我们可以不把线性模型用于$\boldsymbol{x}$本身，而是用在一个变换后的输入$\phi(\boldsymbol{x})$上。深度学习的策略是学习$\phi$。在这种方法中，我们使用的模型为$y = f(\boldsymbol{x}; \boldsymbol{\theta}, \boldsymbol{w})$，其中$\boldsymbol{\theta}$用于学习$\phi$，$\boldsymbol{w}$用于将$\phi(\boldsymbol{x})$映射到所需的输出。

本章大纲：

* 训练前馈网络的设计决策：优化模型、代价函数、输出单元形式
* 隐藏层和**激活函数**（activation function）
* 网络结构
* **反向传播**（back propagation）算法
