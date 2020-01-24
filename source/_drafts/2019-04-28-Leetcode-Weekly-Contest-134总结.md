---
title: Leetcode Weekly Contest 134总结
urlname: leetcode-weekly-contest-134
toc: true
date: 2019-04-28 20:05:43
updated: 2019-04-28 20:05:43
tags: [Leetcode]
categories: Leetcode
---

这周的比赛题意都有些谜……

<!--more-->

## [1033. Moving Stones Until Consecutive](https://leetcode.com/problems/moving-stones-until-consecutive/description/)

标记难度：Easy

提交次数：1/2

代码效率：4ms

### 题意

一条直线上放着三块石头，坐标分别是`a, b, c`。按照如下规则移动石头：假如石头正位于坐标`x < y < z`，则将`x`或`z`位置的石头拿起来，放到`k`处，满足`x < k < z`且`k != y`。问：到石头无法移动时，最多移动过多少次石头，最少移动过多少次石头？

### 分析

这道题的描述很容易让人一看就蒙了。

### 代码

```cpp
class Solution {
public:
    vector<int> numMovesStones(int a, int b, int c) {
        if (a > b) swap(a, b);
        if (a > c) swap(a, c);
        if (b > c) swap(b, c);
        int maxn = (b - a - 1) + (c - b - 1);
        int minn = 2;
        if (a + 1 == b && b + 1 == c) minn = 0;
        else if (a + 1 == b || b + 1 == c) minn = 1;
        else if (a + 2 == b || b + 2 == c) minn = 1;
        return {minn, maxn};
    }
};
```
