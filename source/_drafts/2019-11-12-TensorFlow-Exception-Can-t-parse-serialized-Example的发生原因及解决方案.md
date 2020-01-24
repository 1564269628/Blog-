---
title: 'TensorFlow Exception: Can''t parse serialized Example的发生原因及解决方案'
urlname: tensorFlow-xception-can-t-parse-serialized-example
toc: true
date: 2019-11-12 10:21:41
updated: 2019-11-12 10:21:41
tags: TensorFlow
categories: 深度学习
---

据说[TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord)是很适合TensorFlow的一种数据存储格式。最近我有一些特殊的存储中间训练结果的需求，直接用numpy会有一定的问题，但TFRecord也花了我很长时间去搞定，不得不说TF真是缺文档。

<!--more-->

好久没写博客了，一部分原因是知乎编辑系统太好用了……

## 需求

简单来说，我的需求是：先把10w个句子输入到一个encoder里，得到它们的representation，存到硬盘里备用；然后再把这些representation拿出来作为训练数据，而且最好是以Dataset的形式。当然，这需求很蠢，应该每次直接从encoder里跑一批训练数据比较好；但目前这么写会比较简单。

最简单的处理方法显然是把这堆representation（每个都是句子长度\*512的矩阵）塞到一个npz文件里。但这样做的问题是，文件会很大（8G），而且读入的时候很难以Dataset的形式处理。按照[TF文档](https://www.tensorflow.org/guide/data#consuming_numpy_arrays)的说法：

>Note: The above code snippet will embed the features and labels arrays in your TensorFlow graph as tf.constant() operations. This works well for a small dataset, but wastes memory---because the contents of the array will be copied multiple times---and can run into the 2GB limit for the tf.GraphDef protocol buffer.

把一个numpy数组读进来会导致整个数组变成计算流图里的一个常量结点，于是图可能会爆掉，这就非常不好。而TFRecord似乎就没有这个问题（虽然我不知道是什么机制）。

## 写TFRecord

虽然官网上提供了写TFRecord的指南，但对我并没有什么帮助（没看懂），特别是有存储不定长矩阵需求的时候（事实上我根本没找到矩阵怎么写）。
