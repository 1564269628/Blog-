---
title: 2019-09-08-Leetcode Weekly Contest 153总结
urlname: leetcode-weekly-contest-153
toc: true
date: 2019-09-09 10:40:26
updated: 2019-09-09 10:40:26
tags: 
categories: Leetcode
---

描述

<!--more-->

## [1184. Distance Between Bus Stops](https://leetcode.com/problems/distance-between-bus-stops/description/)

标记难度：Easy

提交次数：1/1

代码效率：8ms

### 题意

有一个多边形，给定多边形每条边的边长，问多边形上某两点之间的最短距离。

### 分析

直接枚举两种可能情况即可。

### 代码

```cpp
class Solution {
public:
    int distanceBetweenBusStops(vector<int>& distance, int start, int destination) {
        if (start > destination) swap(start, destination);
        
        int sum = 0, d1 = 0;
        for (int i = 0; i < distance.size(); i++) {
            sum += distance[i];
            if (start <= i && i < destination)
                d1 += distance[i];
        }
        
        return min(d1, sum - d1);
    }
};
```

## [1185. Day of the Week](https://leetcode.com/problems/day-of-the-week/description/)

标记难度：Easy

提交次数：1/1

代码效率：0ms

### 题意

求给定日期是星期几。

### 分析

已知1970年1月1日是周四（好吧，这个他们给到题目里会更好），那么直接一天天往下推就可以了。注意闰年和每月的天数。

最近Leetcode出的这一类题的确有点多，对于Python调库选手更加友好一些。

### 代码

```cpp
class Solution {
public:
    string dayOfTheWeek(int day, int month, int year) {
        // 1970-1-1 = Thursday
        int start = 3;
        int sum = 0;
        for (int i = 1970; i < year; i++) {
            if (i % 4 == 0 && i % 100 != 0 || i % 400 == 0) {
                sum += 366;
            }
            else
                sum += 365;
        }
        for (int i = 1; i < month; i++) {
            if (i == 1 || i == 3 || i == 5 || i == 7 || i == 8 || i == 10 || i == 12)
                sum += 31;
            else if (i == 2) {
                if (year % 4 == 0 && year % 100 != 0 || year % 400 == 0)
                    sum += 29;
                else
                    sum += 28;
            }
            else
                sum += 30;
        }
        sum += day - 1;
        
        sum = (sum + start) % 7;
        string values[] = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"};
        return values[sum];
    }
};
```

## [1186. Maximum Subarray Sum with One Deletion](https://leetcode.com/problems/maximum-subarray-sum-with-one-deletion/description/)

标记难度：Medium

提交次数：1/1

代码效率：44ms

### 题意

如果**允许**（而不是一定）从数组中删除一个元素，问得到的最大连续子序列和是多少。

### 分析

### 代码

```cpp
class Solution {
public:
    int maximumSum(vector<int>& arr) {
        int maxEnd[100005], maxStart[100005];
        int n = arr.size();
        
        int ans = arr[0];
        for (int i = 0; i < n; i++) {
            maxEnd[i] = arr[i];
            if (i > 0 && maxEnd[i-1] > 0)
                maxEnd[i] = max(maxEnd[i], arr[i] + maxEnd[i-1]);
            ans = max(ans, maxEnd[i]);
        }
        for (int i = n - 1; i >= 0; i--) {
            maxStart[i] = arr[i];
            if (i < n - 1 && maxStart[i+1] > 0)
                maxStart[i] = max(maxStart[i], arr[i] + maxStart[i+1]);
        }

        for (int i = 1; i < n - 1; i++) {
            ans = max(ans, maxEnd[i-1] + maxStart[i+1]);
        }
        
        return ans;
    }
};
```

## [1187. Make Array Strictly Increasing](https://leetcode.com/problems/make-array-strictly-increasing/description/)

标记难度：Hard

提交次数：1/1

代码效率：

* 愚蠢的做法：1876ms

### 题意

### 分析

### 代码

#### 愚蠢的做法

```cpp
class Solution {
    
public:
    int makeArrayIncreasing(vector<int>& arr1, vector<int>& arr2) {
        unordered_map<int, int> mmap;
        sort(arr2.begin(), arr2.end());
        int n = arr1.size(), m = arr2.size();
        mmap[arr1[0]] = 0;
        for (int i = 0; i < m; i++) {
            if (mmap.find(arr2[i]) == mmap.end())
                mmap[arr2[i]] = 1;
        }
        
        for (int i = 1; i < n; i++) {
            unordered_map<int, int> newMap;
            for (auto p: mmap) {
                int key = p.first, value = p.second;
                int newKey = -1, newValue = -1;
                if (key < arr1[i]) {
                    newKey = arr1[i];
                    newValue = value;
                    
                    if (newMap.find(newKey) == newMap.end())
                        newMap[newKey] = newValue;
                    else
                        newMap[newKey] = min(newMap[newKey], newValue);
                }
                
                auto iter = upper_bound(arr2.begin(), arr2.end(), key);
                if (iter != arr2.end()) {
                    newKey = *iter;
                    newValue = value + 1;
                    if (newMap.find(newKey) == newMap.end())
                        newMap[newKey] = newValue;
                    else
                        newMap[newKey] = min(newMap[newKey], newValue);
                }
                
                if (newKey == -1) continue;
                if (newMap.find(newKey) == newMap.end())
                    newMap[newKey] = newValue;
                else
                    newMap[newKey] = min(newMap[newKey], newValue);
            }
            mmap = newMap;
        }
        
        int ans = 1e9;
        if (mmap.size() == 0) return -1;
        for (auto p: mmap)
            ans = min(ans, p.second);
        return ans;
    }
};
```