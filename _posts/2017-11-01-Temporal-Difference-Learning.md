---
layout: post
title: Temporal-Difference Learning
tags: [machine-learning, reinforcement-learning, sutton-book]
date: 2017-11-03 02:00:00 +08:00
---

TD learning是对Monte Carlo和DP算法的综合，像Monte Carlo一样属于一种Modle Free算法，即不需要对环境模型有一个具体的认识（比如需要知道状态转移的Markov决策过程）；同时也像DP一样属于一种迭代更新算法。也许这个表达式可以更容易地帮助理解：

$$V(S_t) \leftarrow V(S_t) + \alpha[R+\gamma V(S_{t+1})-V(S_t)]$$

其中 $V(S_t)$ 表示在 $t$ 时刻状态为 $S$ 的 state value，在Monte Carlo里面，$V(S)$ 的估计通常采用 *first visit* 进行，以保证估计过程的 *unbias* 特性。从公式角度来看，和Monte Carlo唯一的区别就是把原来的 $G_t$ 改成了基于下一时刻 state value 的估计 $V(S_{t+1})$ 得到的 $V'(S_t)$。下面是对TD(0)的算法描述：

<img src="/assets/images/td0.png" width="50%"/>

这里有这么一个问题，虽然从递归的性质来看，逻辑关系没有错误，但是所有的 $V(S_t)$ 的预测都是基于对 $V(S_{t+1})$ 预测的预测，假设对 $V(S_{t+1})$ 的预测有bias，怎么就能保证 $V(S_t)$ 的预测的偏差不会出现累积效应，或者说越来越偏离真实值？

**TD算法的优点**

TD算法和DP以及Monte Carlo相比，首先的优点就是计算空间小，达到optimal的时间短，以及model-free。尤其是在有些长episode场景中，由于Monte Carlo需要在每个episode结束的时候才开始进行state value或者action value的更新，那么达到收敛的时间花费显然要高，相比TD的step by step更新就很cost。

当然到这里为止还是没有解决上面那个问题，实际试验结果显示TD确实能够像Monte Carlo那样收敛到最优值，但没有给出实际的数学证明（所以这里先埋个坑，过段时间再来填）

**TD(0)的最优性**

在学习率 $\alpha$ 的值足够小的情况下，batch TD能够收敛一个最优值，同时不依赖于 $\alpha$ 的取值，在相同的条件下，Monte Carlo也可以收敛到一个最优值，当然这两个值不一定相同。因为两者的优化目标是不一样的，Monte Carlo通过做出和训练集相比均方误差最小的值估计来逼近最优，而TD算法是通过对Markov决策的最大似然估计来逼近最优解。举个例子：这里有8组episode的抽样模拟结果：

<img src="/assets/images/example64.png" width="50%"/>

接下来分别使用TD和Monte Carlo来对两种状态的 $V$ 进行估计，显然 $V(B)$ 的值很好算，在两种算法的情况下都是 $3 \over 4$，而对状态 $A$ 的state value估计却有两种情况，在Monte Carlo里面，$V(A)=0$，在TD里面 $V(A)={3 \over 4}$，然而实际的状态转移情况是：

<img src="/assets/images/example642.png" width="50%"/>

两种估值都没有错，因为都能够控制从A走到B，Monte Carlo保证了在训练集上的error最小（实际上这里没有误差），而TD虽然在训练集上的误差要比Monte Carlo大，但是在将来的情况表现得会比Monte Carlo的预测要好。

这里接入有关TD(0)的几种算法

**Sarsa：On-policy TD(0) control**

Sarsa在保证epsiode的时间有限的情况下，能够以1.0的概率收敛到最优的policy和action value

<img src="/assets/images/sarsa_td.png" width="50%"/>

**Q-learning：Off-policy TD(0) control**

<img src="/assets/images/q_learning_td.png" width="50%" />

Q-learning和Sarsa的不同点可以这样来理解，前者总是通过选取action value最大的动作来执行，后者通过可能通过一个更安全的动作来执行，因为每次动作的选取都是基于policy的。

**Expected Sarsa**

和普通的Sarsa相比，该算法的特点是弱化了由于policy的random特性而导致的方差过大，同时计算量也上升了

$$Q(S_t, A_t) \leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma E[Q(S_{t+1},A_{t+1}) \mid S_{t+1}]-Q(S_t,A_t)]$$

$$Q(S_t, A_t) \leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma \sum_a{\pi(a\mid S_{t+1})Q(S_{t+1},A_{t+1})}-Q(S_t,A_t)]$$

一般的Q-learning有一个缺点，假设现在状态 $S$ 的最优action value真实值是0，而它目前的所有action估计值有正有负有零，从其预测更新的规则来看，每次都是选取最大的预测值，那么这样一来就会导致整体出现一个正向bias。有一种解决方法，通过将动作预测和执行分开来做，能够保证这个过程是无偏估计。

**Double Q-learning**

假设现在有两套policy的参数，一套 $Q_1$ 用来进行action预测，一套 $Q_2$ 用来进行动作的执行，也就是进行当前state的action value更新，那么假设用来更新的动作 $A^*=\arg \max_a Q_1(a)$，那么

$$Q(s,a)=Q(s,a)+\alpha[R+\gamma Q_2(s',\arg \max_aQ_1(s',a))-Q(s,a)]$$

还要说明一下的是其中一个 $Q$ 是作为target来考虑的，也就是其中一个被当作最新的policy，用来产生 $A^*$ 的那套参数便是最新的。那么可以这样来理解，我用最新的一套policy产生了一个action，而这个action由于迭代关系又是与上一次的policy有关的，也就相当于满足老的policy的分布，而这个分布暂时还在另一个Q里面，所以现在把这个action拿给另一个Q来预测更新就是无偏的。

<img src="/assets/images/double_q.png" width="50%"/>