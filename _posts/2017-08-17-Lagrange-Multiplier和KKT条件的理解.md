---
layout: post
title: Lagrange Multiplier和KKT条件的理解
tags: [mathematics, machine-learning]
date: 2017-08-17 18:55:00 +08:00
---

拉格朗日乘数法和KKT条件在凸优化问题中的作用很大，当目标函数属于凸函数问题时，使用这两种方法能够有效且简单的得到最优值。当然，两种方法也不是万能的，建立在数学上成立的算法应用到实际还是需要考虑机器效率，SVM使用KKT添加得到 $\alpha$ 集合条件但还需要使用SMO算法优化。

### 凸优化问题的分类

> 在求解凸优化问题过程中，一个关键的想法是**所有的局部最优都是全局最优**

1. 线性规划（*Linear Programming*）：如果优化问题的目标函数 $f$ 和以及不等式约束 $g_i$ 都是仿射函数。

2. 二次规划（*Quadratic Programming*）：如果优化问题的目标函数 $f$ 变量的次数最高为2，而所有的不等式约束条件都是仿射函数。

3. 二次约束二次规划（*Quadratically Constrained Quadratic Programming*）：如果约束条件和目标函数的变量次数最高都为2。

4. 半定规划（*Semidefinite Programming*）：这里我们使用矩阵方式来描述优化问题，比如优化的目标函数是：$\min tr(CX)$，约束条件：

   $s.t. tr(A_iX) = b_i, i = 1, …, p; X \ge 0$

   其中 $C, A_1, …, A_p \in \S^n$ 为对称矩阵。

### Lagrange Multiplier如何解决凸优化问题呢？

引入wikipedia上的一个例子，假设我们的目标函数以及约束条件为：

$$\min f(x, y)$$

$$s.t. g(x, y) = c$$

wikipedia上使用了这么一幅图来具象化这个优化问题：

<img src="/assets/images/lagrange.png" width="40%"/>

当目标函数取得最优值时，约束条件函数和目标函数等高线的法线平行，表示接触点相切。这是当然如果仅仅是相交的话，我们没有办法保证接触点是最优值，因为在 $max$ 问题中的交点会小于最优值，在 $min$ 问题中的交点会大于最优值。

这种情况可以通过以下形式来描述：

$$\nabla f(x,y) = \nabla -\lambda g(x, y) ·············· (1)$$

因为原问题是 $\min f(x,y)$，为了保持一致性，使用拉格朗日乘数法表示为：

$$\min L(x, y, \lambda) = f(x, y) + \lambda(g(x, y) - c) ·············· (2)$$

$$s.t. \lambda \ge 0; g(x,y) - c = 0$$

联立（1）和（2），可知：$\nabla L(x, y, \lambda) = 0$，即是最优值出现的条件


### KKT又是如何解决凸优化问题呢？

上面可以看到，在普通的拉格朗日乘数法中，约束条件都是都是等式约束，但是实际问题可能也包含不等式约束，而这种优化问题的解决就需要KKT条件来辅助。需要说明的是：**对于一般问题，KKT条件是一组使得解成为最优解的必要条件，而只有当原问题是凸优化问题时，KKT才成为充要条件；KKT条件也是作用在强对偶问题才有效的条件**

*怎么才叫强对偶？*

强对偶关系指的是原问题的最优解和对偶问题的最优解一致，然而这一点却是默认包含在KKT中的（后面的推导有体现）

假设目标函数为：

$$L(x, \mu) = f(x) + \sum_{i=1}^m \mu_kg_k(x)$$

$$s.t. \mu_k \ge 0; g_k(x) \le 0$$

即有：$\mu_kg_k(x) \le 0$,

那么：$\max_\mu L(x, \mu) = f(x)$，

同样的有：$\min_xf(x) = \min_x\max_\mu L(x, u)········(3)$

再看对偶问题：

$$\max_\mu\min_xL(x, \mu) = \min_x f(x) + \max_\mu\min_x \mu g(x)$$

因为 $\mu \ge 0$，且 $g(x) \le 0$，那么：$\min_x \mu g(x) = 0$ 当 $\mu = 0$ 或者 $g(x) = 0$时成立；$\min_x \mu g(x) = -\infty$ 当 $\mu > 0$ 而且 $g(x) \le 0$ 时成立。

显然，$\max_\mu\min_x\mu g(x) = 0$，即：

$$\max_\mu\min_xL(x, \mu) = \min_x f(x) + \max_\mu\min_x\mu g(x) = \min_x f(x)··········(4)$$

此时有：$\mu = 0$ 或者 $g(x) = 0$

联立（3）和（4）有：$\min_x\max_\mu L(x, \mu) = \max_\mu\min_x L(x, \mu)$

这里便说明了强对偶性。也就是说原问题和对偶问题的最优解都在：

**$\mu=0$ 或者 $g(x)=0$** 处存在。



### SVM中的应用

[在编中]



