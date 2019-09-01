---
title: Leetcode Weekly Contest 152总结
urlname: leetcode-weekly-contest-152
toc: true
date: 2019-09-01 11:11:04
updated: 2019-09-01 11:11:04
tags: [Leetcode, Leetcode Contest, alg:Math]
categories: Leetcode
---

这周的比赛比较简单。

<!--more-->

## [1175. Prime Arrangements](https://leetcode.com/problems/prime-arrangements/description/)

标记难度：Easy

提交次数：1/1

代码效率：4ms

### 题意

将1到`n`进行排列，使得素数在素的index上，问有多少种排列方法。

### 分析

首先把素数都找出来，然后分别计算素数和非素数的排列，再乘起来即可。

### 代码

```cpp
class Solution {
    bool isPrime[102];
    int prime[102];
    int m;
    
    // initialize prime table
    void init() {
        m = 0;
        memset(isPrime, 1, sizeof(isPrime));
        isPrime[1] = false;
        isPrime[2] = true;
        for (int i = 2; i <= 100; i++) {
            if (!isPrime[i]) continue;
            prime[m++] = i;
            for (int j = i + i; j <= 100; j += i)
                isPrime[j] = false;
        }
    }
    
    typedef long long int LL;
    LL P = 1e9 + 7;
    
    LL calcA(int x) {
        LL prod = 1;
        while (x) {
            prod = (prod * x) % P;
            x--;
        }
        return prod;
    }
    
public:
    int numPrimeArrangements(int n) {
        init();
        int prime_cnt = 0;
        for (int i = 1; i <= n; i++)
            if (isPrime[i])
                prime_cnt++;
        
        LL ans1 = calcA(prime_cnt), ans2 = calcA(n - prime_cnt);
        return (int) (ans1 * ans2 % P);
    }
};
```

## [1176. Diet Plan Performance](https://leetcode.com/problems/diet-plan-performance/description/)

标记难度：Easy

提交次数：1/1

代码效率：144ms

### 题意

计算每连续`K`天摄入的卡路里量，打分，并计算总评分。

### 分析

水题，直接计算即可。

### 代码

```cpp
class Solution {
public:
    int dietPlanPerformance(vector<int>& calories, int k, int lower, int upper) {
        int sum = 0;
        int score = 0;
        int n = calories.size();
        for (int i = 0; i < n; i++) {
            sum += calories[i];
            if (i >= k - 1) {
                if (i >= k) sum -= calories[i - k];
                if (sum > upper) score++;
                else if (sum < lower) score--;
            }
        }
        return score;
    }
};
```

## [1177. Can Make Palindrome from Substring](https://leetcode.com/problems/can-make-palindrome-from-substring/description/)

标记难度：Medium

提交次数：1/1

代码效率：892ms

### 题意

给定一个字符串`s`和一些查询`[l, r, k]`，问`[l, r]`范围内的字符串能否在替换`k`个字符的范围内重排成一个回文串。

### 分析

首先，我使用了基于倒排列表（inverted list）的方法来统计一个区间内每个字符的出现次数。不过，事实上我们只需要出现次数为奇数的字符个数。

然后可以分成以下几种情况来讨论：记出现次数为奇数的字符个数为`l`。如果`l`为0或1，显然根本就不需要替换任何字符，直接就可以重新排列成回文数。而将一对出现次数为奇数的字符中的一个替换成另一个可以使这两种字符的出现次数均发生变化，因此最多可以改变`2*k`种字符的出现次数，因此`l <= 2 * k`或`l - 1 <= 2 * k`时可以重排，总的来说就是`l <= 2 * k + 1`。

### 代码

```cpp
class Solution {
    unordered_map<char, vector<int>> idxMap;
    
    int countInvList(char c, int l, int r) {
        auto iter = idxMap.find(c);
        if (iter == idxMap.end()) return 0;
        auto up_iter = upper_bound(iter->second.begin(), iter->second.end(), r);
        auto lo_iter = lower_bound(iter->second.begin(), iter->second.end(), l);
        int cnt = up_iter - lo_iter;
        return cnt;
    }
    
public:
    vector<bool> canMakePaliQueries(string s, vector<vector<int>>& queries) {
        for (int i = 0; i < s.length(); i++)
            idxMap[s[i]].push_back(i);
        
        vector<bool> ans;
        
        int oddCnt;
        for (vector<int>& q: queries) {
            int l = q[0], r = q[1], k = q[2];
            
            oddCnt = 0;
            for (int i = 0; i < 26; i++) {
                char c = (char) (i + 'a');
                int cnt = countInvList(c, l, r);
                if (cnt % 2 == 1)
                    oddCnt++;
            }
            
            bool a = false;
            if (oddCnt <= k * 2 + 1) a = true;
            ans.push_back(a);
        }
        
        return ans;
    }
};
```

## [1178. Number of Valid Words for Each Puzzle](https://leetcode.com/problems/number-of-valid-words-for-each-puzzle/description/)

标记难度：Hard

提交次数：1/1

代码效率：552ms

### 题意

给定一个词表`words`和一个query表`puzzles`，一个`word`在一个`puzzle`的范围内仅当满足以下条件：

* `word`包含`puzzle`的首字母
* `puzzle`包含`word`的每一个字母

问每个`puzzle`包含多少个`word`。其中`len(puzzle) == 7`。

### 分析

使用位运算来判断一对`(word, puzzle)`是否合法是平凡的，但`O(N*M)`的复杂度是不可接受的。仔细考虑使用位运算来判断的过程，并观察到`len(puzzle) == 7`，可以发现，通过`puzzle`的bit mask生成所有合法的`word`的bit mask的代价并不是太高。

举例说明，当`puzzle = "bac"`时，合法的`words = ["b", "ab", "bc", "abc"]`。当`puzzle`中的每个字符都不重复时，最多有`2^6=128`个这样的`word`。

于是可以得到算法：

* 用哈希表（`map`）统计所有`word`的bit mask
* 对于每一个`puzzle`，生成所有合法的`word`的bit mask，计算其中存在的`word`的总数

### 代码

```cpp
class Solution {
    int conv2mask(string s) {
        int mask = 0;
        for (char c: s)
            mask |= 1 << (c - 'a');
        return mask;
    }
    
    vector<int> genPuzzleMask(string s) {
        char f = s[0];
        // 排序和去重
        sort(s.begin(), s.end());
        vector<char> chs;
        for (char c: s)
            if (chs.empty() || c != chs.back())
                chs.push_back(c);
        
        // 生成对应关系
        int n = chs.size();
        vector<int> gen;
        for (int i = 0; i < (1 << n); i++) {
            int realMask = 0;
            for (int j = 0; j < n; j++) {
                if (i & (1 << j))
                    realMask |= 1 << (chs[j] - 'a');
            }
            // 注意首字母的限制
            if (realMask & (1 << (f - 'a')))
                gen.push_back(realMask);
        }
        return gen;
    }
    
public:
    vector<int> findNumOfValidWords(vector<string>& words, vector<string>& puzzles) {
        map<int, int> valPuz;
        for (string s: words) {
            int mask = conv2mask(s);
            valPuz[mask]++;
        }
        
        vector<int> ans;
        for (string s: puzzles) {
            vector<int> masks = genPuzzleMask(s);
            int b = 0;
            for (int m: masks)
                b += valPuz[m];
            ans.push_back(b);
        }
        return ans;
    }
};
```
