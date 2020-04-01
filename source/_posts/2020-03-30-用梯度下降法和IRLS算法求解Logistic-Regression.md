---
title: 用梯度上升法和IRLS算法训练Logistic Regression模型
urlname: logistic-regression-gd-irls
toc: true
mathjax: true
date: 2020-03-30 02:59:30
updated: 2020-03-30 02:59:30
tags: Machine Learning
categories: 机器学习
---

本文主要介绍了以下内容：

* 什么是Logistic Regression
* 如何用梯度上升法训练Logistic Regression
* 如何用IRLS算法训练Logistic Regression

<!--more-->

## 什么是Logistic Regression

Logistic Regression是一种（二）分类器，形式为

$$
P(y=1 | \boldsymbol{x}) = \frac{1}{1 + \exp{(-(w_0 + \boldsymbol{w}^T \boldsymbol{x}))}}
$$

这种建模方式是为了用线性回归解决分类问题。如果直接令$P(y=1 | \boldsymbol{x}) = w_0 + \boldsymbol{w}^T \boldsymbol{x}$，由于线性函数是无界的，显然上述概率分布不合法。因此对$P(y=1 | \boldsymbol{x})$进行logistic变换，使得

$$
\log{\frac{P(y=1 | \boldsymbol{x})}{1 - P(y=1 | \boldsymbol{x})}} = w_0 + \boldsymbol{w}^T \boldsymbol{x}
$$

求解即得到上述形式。

在实际预测中，当$w_0 + \boldsymbol{w}^T \boldsymbol{x} > 0.5$时（即$w_0 + \boldsymbol{w}^T \boldsymbol{x} < 0$时），预测$y=1$，否则预测$y = 0$，因此Logistic Regression是一个线性分类器。

为简单起见，令$x_0 = 1$，$\boldsymbol{w} = [w_0, \boldsymbol{w}]$，以下将分类器简写为

$$
P(y=1 | \boldsymbol{x}) = \frac{1}{1 + \exp{(-\boldsymbol{w}^T \boldsymbol{x})}}
$$

## Logistic Regression的似然函数

给定训练数据$\{(\boldsymbol{x_i}, y_i)\}_{i=1}^N$，由于只有$P(Y|X)$，没有$P(X)$或$P(X|Y)$，因此无法进行最大似然估计，只能进行最大条件似然估计（Maximum Conditional Likelihood Estimate）：

$$
\hat{\boldsymbol{w}}=\underset{\boldsymbol{w}}{\operatorname{argmax}} \prod_{i=1}^{N} P\left(y_{i} | \boldsymbol{x}_{i}; \boldsymbol{w}\right)
$$

记{% raw %}$p(\boldsymbol{x}_i) = P(y_{i} = 1 | \boldsymbol{x}_{i}; \boldsymbol{w})${% endraw %}，则

{% raw %}
$$
\begin{aligned}
\mathcal{L}(\boldsymbol{w}) &= \log{\prod_{i=1}^N P(y_{i} | \boldsymbol{x}_{i}; \boldsymbol{w})} \\
&= \log{\prod_{i=1}^N p(\boldsymbol{x}_i)^{y_i} (1 - p(\boldsymbol{x}_i))^{1-y_i}} \\
&= \sum_{i=1}^N y_i \log{p(\boldsymbol{x}_i)} + (1-y_i)\log{(1 - p(\boldsymbol{x}_i))} \\
&= \sum_{i=1}^N y_i \log{\frac{p(\boldsymbol{x}_i)}{1 - p(\boldsymbol{x}_i)}} + \log{(1 - p(\boldsymbol{x}_i))} \\
&= \sum_{i=1}^N y_i(\boldsymbol{w}^T \boldsymbol{x}_i) + \log{\frac{\exp{(-\boldsymbol{w}^T \boldsymbol{x}_i)}}{1 + \exp{(-\boldsymbol{w}^T \boldsymbol{x}_i)}}} \\
&= \sum_{i=1}^N y_i(\boldsymbol{w}^T \boldsymbol{x}_i) + \log{\frac{1}{1 + \exp{\boldsymbol{w}^T \boldsymbol{x}_i}}} \\
&= \sum_{i=1}^N y_i(\boldsymbol{w}^T \boldsymbol{x}_i) - \log{(1 + \exp{(\boldsymbol{w}^T \boldsymbol{x}_i)})}
\end{aligned}
$$
{% endraw %}

将似然函数对$\boldsymbol{w}$求导，得到

{% raw %}
$$
\left.\frac{\partial \mathcal{L}(\boldsymbol{w})}{\partial w_j} \right|_{\boldsymbol{w}_t} = \sum_{i=1}^N y_i x_{ij} - \mu_i^t x_{ij} = \sum_{i=1}^N x_{ij} (y_i - \mu_i^t)
$$
{% endraw %}

其中

{% raw %}
$$
\mu_i^t = \frac{1}{1 + \exp{(-\boldsymbol{w}^T \boldsymbol{x}_i)}} = P(y_{i} = 1 | \boldsymbol{x}_{i}; \boldsymbol{w}_t)
$$
{% endraw %}

则有

{% raw %}
$$
\begin{aligned}
\nabla_{\boldsymbol{w}} \mathcal{L}(\boldsymbol{w}) |_{\boldsymbol{w}_t} &= \left.\left[\frac{\partial \mathcal{L}}{\partial w_0}, \frac{\partial \mathcal{L}}{\partial w_1}, \cdots, \frac{\partial \mathcal{L}}{\partial w_d}\right]^T \right|_{\boldsymbol{w}_t} \\
&= \sum_{i=1}^N \boldsymbol{x}_i (y_i - \mu_i^t)
\end{aligned}
$$
{% endraw %}

## 用梯度上升法训练Logistic Regression

得到似然函数对$\boldsymbol{w}$的梯度之后，即可立即得到梯度上升法的更新函数：

{% raw %}
$$
\boldsymbol{w}_{t+1} = \boldsymbol{w}_t + \eta \nabla_{\boldsymbol{w}} \mathcal{L}|_{\boldsymbol{w}_t} = \boldsymbol{w}_t + \eta \sum_{i=1}^N \boldsymbol{x}_i (y_i - \mu_i^t)
$$
{% endraw %}

由于$\mathcal{L}$是凸函数，因此梯度上升法总能收敛。

## 用IRLS（Iterative reweighted least squares)法求解

牛顿法的迭代公式为

{% raw %}
$$
x_{t+1} = x_t - \frac{f(x_t)}{f'(x_t)}
$$
{% endraw %}

推广到Logistic Regression，可以得到

{% raw %}
$$
\boldsymbol{w}_{t+1} \leftarrow \boldsymbol{w}_{t}-\left.H^{-1} \nabla_{\boldsymbol{w}} \mathcal{L}(\boldsymbol{w})\right|_{\boldsymbol{w}_{t}}
$$
{% endraw %}

其中$H$是Hessian矩阵：

{% raw %}
$$
H = \left.\nabla_{\boldsymbol{w}}^2 \mathcal{L}(\boldsymbol{w})\right|_{\boldsymbol{w}_{t}}
$$
{% endraw %}

下面推导具体的更新式子。

{% raw %}
$$
\begin{aligned}
\nabla_{\boldsymbol{w}} \mathcal{L}(\boldsymbol{w}) |_{\boldsymbol{w}_t} &= \sum_{i=1}^N \boldsymbol{x}_i (y_i - \mu_i^t) \\
& = X (\boldsymbol{y} - \boldsymbol{\mu}^t)
\end{aligned}
$$
{% endraw %}

其中$X = [\boldsymbol{x_1}, \cdots, \boldsymbol{x_N}]$，$\boldsymbol{y} = [y_1, \cdots, y_N]^T$，$\boldsymbol{\mu}^t = [\mu_1^t, \cdots, \mu_N^t]^T$。

{% raw %}
$$
\begin{aligned}
\left.\frac{\partial^2 \mathcal{L}(\boldsymbol{w})}{\partial w_{i_1} \partial w_{i_2}}\right|_{\boldsymbol{w}_t} &= \frac{\partial \mathcal{L}(\boldsymbol{w})}{\partial w_{i_2}} \sum_{i=1}^N x_{ii_1} (y_i - \mu_i^t) \\
&= -\sum_{i=1}^N x_{ii_1} x_{ii_2} \mu_i^t (1 - \mu_i^t)
\end{aligned}
$$
{% endraw %}

即

{% raw %}
$$
\begin{aligned}
H &= -\sum_{i=1}^N \mu_i^t (1 - \mu_i^t) \boldsymbol{x}_i \boldsymbol{x}_i^T \\
&= -XR^tX^T
\end{aligned}
$$
{% endraw %}

其中$R^t$是对角阵，$R_{ii}^t = \mu_i^t (1 - \mu_i^t)$。

最后推导出$\boldsymbol{w}$的更新公式：

{% raw %}
$$
\begin{aligned}
\boldsymbol{w}_{t+1} &= \boldsymbol{w}_{t}-\left.H^{-1} \nabla_{\boldsymbol{w}} \mathcal{L}(\boldsymbol{w})\right|_{\boldsymbol{w}_{t}} \\
&= \boldsymbol{w}_{t} - \left(-XR^tX^T\right)^{-1} X (\boldsymbol{y} - \boldsymbol{\mu}^t) \\
&= \boldsymbol{w}_{t} - \left(XR^tX^T\right)^{-1} X (\boldsymbol{\mu}^t - \boldsymbol{y}) \\
&= \left(XR^tX^T\right)^{-1} \left[XR^tX^T\boldsymbol{w}_{t} - X (\boldsymbol{\mu}^t - \boldsymbol{y})\right] \\
&= \left(XR^tX^T\right)^{-1} XR^t\boldsymbol{z}
\end{aligned}
$$
{% endraw %}

其中

$$
\boldsymbol{z} = X^T\boldsymbol{w}_{t} - (R^t)^{-1} (\boldsymbol{\mu}^t - \boldsymbol{y})
$$

### 增加正则化项

增加正则化项后，对数似然函数变为

$$
-\frac{\lambda}{2} \|\boldsymbol{w}\|^2 + \mathcal{L}(\boldsymbol{w})
$$

此时

{% raw %}
$$
\begin{aligned}
\nabla_{\boldsymbol{w}} \mathcal{L}(\boldsymbol{w}) |_{\boldsymbol{w}_t} &= X (\boldsymbol{y} - \boldsymbol{\mu}^t) - \lambda \boldsymbol{w} \\
H = \left.\nabla_{\boldsymbol{w}}^2 \mathcal{L}(\boldsymbol{w})\right|_{\boldsymbol{w}_{t}} &= -XR^tX^T - \lambda I
\end{aligned}
$$
{% endraw %}

## 代码

代码见[logistic-regression](https://github.com/zhanghuimeng/logistic-regression)，使用的数据集是[UCI a9a](http://ml.cs.tsinghua.edu.cn/~wenbo/data/a9a.zip)，实现了梯度上升法和IRLS算法。算法的具体使用方法和运行结果见README。需要注意的几点是：

* 梯度上升法对初始值不敏感，但IRLS对初始值敏感，$w$的绝对值不能太大
* 由于数据的稀疏性，IRLS求Hessian矩阵时可能会出现奇异矩阵，此时可以用伪逆（`np.linalg.pinv`）来代替逆，也可以增加正则项使得Hessian矩阵不再奇异

## 参考文献

感谢助教和sls与我的讨论。

* [14. Logistic Regression and Newton’s Method](http://www.stat.cmu.edu/~cshalizi/402/lectures/14-logistic-regression/lecture-14.pdf)
