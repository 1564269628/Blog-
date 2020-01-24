---
title: Codeforces 1183 H（DP）
urlname: codeforces-1183-h
toc: true
date: 2019-06-28 00:47:05
updated: 2019-06-28 00:47:05
tags: [Codeforces, Codeforces Contest, alg:DP]
categories: Codeforces
---

题目：[https://codeforces.com/contest/1183/problem/H](https://codeforces.com/contest/1183/problem/H)

代码：[https://codeforces.com/contest/1183/submission/56171157](https://codeforces.com/contest/1183/submission/56171157)

昨天刚刚打完的一场Div3的比赛，比赛中当然是没有做出来这道题了（甚至以为应该用后缀数组来做）。后来想了想，才觉得应该是DP，而且是高维DP。（有这个分类吗？）

<!--more-->

## 题意

给定一个字符串，求长度为1-n的不重复子序列的数量，然后判断总数是否小于`k`，如果小于，则求一个（比较平凡的）代价。

## 分析

### 我的做法

显然重点是求各个长度的不重复子序列的数量。我是这么想的：首先用`f[i][j][l]`表示`s[i..j]`中有多少个不重复的子序列。于是很容易得到一个递推式：`f[i][j][l] = f[i][j-1][l-1] + f[i][j-1][l] - 重复`。这个式子的意思是，`s[i..j]`中长度为`l`的不重复序列的数量等于`s[i..j-1]`中
