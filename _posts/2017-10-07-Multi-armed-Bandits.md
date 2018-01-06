---
layout: post
title: Multi-armed Bandits
tags: [machine-learning, reinforcement-learning, sutton-book]
date: 2017-10-07 22:44:00 +08:00
---

> 将强化学习和其他学习方式区分开来的一个重要特征是：RL通过评估选取动作而不是指导agent应该执行哪些正确的动作，但其实将这两种方式融合在一起也是很有趣的。

## $k$-armed Bandit问题

比如你需要在 $k$ 个不同的action中重复做出选择，在每个action决定作出后，你都会得到服从固定概率分布的reward，你的目标是最大化整个学习过程中的reward。为了方便后面的描述，我们做出如下定义：

- $A_t$: the action selected on time step $t$
- $R_t$: the corresponding reward of $At$
- $q_*(a)$: the value then of an arbitrary aciton $a$

其中 $q_*(a)=E[R_t \mid A_t=a]$

假设我们事先不知道每个动作的reward积累值，其估计值表示为：$Q_t(a) = q_*(a)$。

如果我们在整个学习过程中都维护这些动作的估计值，便可在每个时刻都拿到当时状态的最大估计，这是一种贪心策略。当agent基于以往的经验选择其中一个action时，将之称为一次实践。当然如果放弃使用这种贪婪方式，比如随机选择一个action，那么我们将其称为一次探索（或者学习），因为这样的选择方式能够更新action对应的Q值。在强化学习中，这样的探索方式是很重要的，我们需要考虑在某个状态下，选择不一样的动作可能导致最后的反馈结果会比之前的最优方式更好。

那么问题来了，我们要如何平衡好这个模型中的学习和实践的时间呢？先来看两种不同的值估计。

**Action-value Methods**

如果通过平均水平来做出action的值估计，那么一个action在t时刻之前奖励的平均估计表示为：

$$Q_t(a) = \frac{t时刻前a动作的奖励和}{t时刻前执行a动作的总次数}  = \frac{\sum_{i=1}^{t-1}R_i \cdot 1_{A_i=a} } {\sum_{i=1}^{t-1}1_{A_i=a} } $$

其中，如果分母是0的话，我们将其定义估计值定义为一个默认值，比方说$Q_1(a)=0$。当分母趋于无限时，根据大数定律，$Q_t(a)$ 将收敛到 $q_*(a)$。

在这基础上，我们接下来需要知道这些动作的估计值是如何反映到动作的选择上去的。最简单的方法当然是每次都选择估计值最大的动作了：$Q_t(A_t^*)=\max_{a}Q_t(a)$。根据这个原则，那么 $t$ 时刻的动作选择就是：

$$A_t=\arg \max_{a}Q_t(a)$$

这种贪心策略选择总是利用agent到目前为止积累的知识来最大限度地立即获得奖励，没有把时间花在尝试更多显然是较差的动作选择上，看看它们是否真的会更好。这里有一个简单的策略：给定一个固定的概率 $\epsilon$，代表随机选择动作的概率，也就是相当于增加了模型学习的机会，我们把这样的方式称为 $\epsilon$-greedy。从概率上讲，这样的方式也近似地保证了所有的 $Q_t(a)$ 都能够收敛到 $q_*(a)$。

来看看贪心方式和 $\epsilon$-greedy 方法的区别

<img src="/assets/images/sb_ch2_greedy.png" width="40%"/>

显然 $\epsilon$-greedy 方法要好于greedy方法，但这是建立在 $q_*(a)$ 的概率分布满足存在一定方差的情况（当然实际情况也是）。如果在某些情况下其分布的方差为0，也就是说，只要使用贪心的方法，每次选择的action都会是保证value最优，在这种情况下，显然greedy要更好了。

**Incremental Implementation**

使用直接使用平均值来进行value的估计，可能会面临一个计算量大的问题，尤其是学习系统中的aciton和时间规模都比较庞大的时候。

假设让 $R_i$ 表示在第 $i$ 次选择某个动作的奖励值，让 $Q_n$ 表示这个动作在被选择 $n-1$次之后的估计值，那么我们可以将 $Q_n$ 表示为：

$$Q_n = { {R_1 + R_2 + … + R_{n - 1} } \over {n - 1} }$$

如果直接这样计算估计值，那么我们需要维护关于每个动作的至少 $n$ 个不同的激励值，我们可以做出如下化简：

$$Q_{n + 1} = { 1 \over n } \sum_{i=1}^nR_i={1 \over n}(R_n + \sum _{i=1}^{n - 1}R_i)={1 \over n}(R_n + (n-1)Q_n)=Q_n+{1 \over n}[R_n - Q_n]$$

这样的话，我们只需要维护前一次的估计值并记录已经选择过当前动作的次数，我们可以对上式进行进一步的抽象化（更新估计值的规则）

$$NewEstimate \leftarrow OldEstimate + StepSize [Target - OldEstimate]$$

那么使用增量式实现的bandit算法伪代码如下（$bandit(a)$ 表示选择某个动作 $a$ 能够得到的reward）

<img src="/assets/images/sb_ch2_simple_bandit.png" width="50%"/>

## 跟踪非稳定问题

以上两种方法在稳定环境状态问题下能够比较好的发挥作用，但我们经常遇见的强化学习问题都属于非稳定环境问题，也就是agent面临的学习场景会时常改变，在这些情况下，我们对即时或者说最近的reward的关注要多于长时的reward。

一个最常用的方法是使用一个常量step-size，也就是下面的 $\alpha$ 参数

$$Q_{n+1}=Q_n + \alpha[R_n-Q_n]$$

其中 $\alpha \in (0,1]$，那么新的 $Q_{n+1}$ 可以改成：

$$Q_{n+1}=(1-\alpha)^nQ_1+\sum_{i=1}^n\alpha(1-\alpha)^{n-i}R_i$$

不难发现其实：$(1-\alpha)^n+\sum_{i=1}^n\alpha(1-\alpha)^{n-i}=1$。从统计学方面讲，考虑到 $1-\alpha$ 值小于1，其指数越大，导致对应的 $R_i$ 的权重越低，而且随着 $n$ 的逐渐增大，离当前动作越远的reward对其估计值的贡献越小，而当前动作对应的reward值的贡献越大，这样便达到了之前要求更多关注最近reward的要求。

有时候逐步改变学习步长是有好处的，因为那样能够保证收敛（虽然不能保证动作集中的所有action对应的估计都收敛）

<center>$\sum_{n=1}^\infty{\alpha_n(a)}=\infty $ and $\sum_{n=1}^\infty{\alpha_n^2(a) \lt \infty}$</center>

当学习步长为常量时，第二个条件不满足，也就是说值估计永远都没法收敛，但是对于最近收到的奖励值，其估计值会因此不断调整。这也就想我们之前提出的问题一样，在变化的学习环境中，我们的估计值显然也应当不停变化。

另外，一系列满足上面收敛条件的学习参数经常会收敛得比较慢，而且还需要考虑进行一定程度的调参工作才能使得最终的结果令人满意。虽然这种方法经常会在理论中出现，但是实际应用中我们几乎不会使用。

## Optimistic Initial Value

之前提到的两种方法，似乎都在一定程度上对初始的动作值估计 $Q_1(a)$ 有一定程度的依赖。但是在样本平均方法中，也就是第一种方法，只要所有的动作至少发生一次，这种初始值估计偏差就会消失，而在第二种方法，也就是保持学习步长为常量的那种方法中，这种偏差会一直存在，即使随着学习时间的延长，这种偏差带来的影响越来越小。

有趣的是，实际情况这种估计偏差其实是很有用的。它的一个负面影响就是动作值的初始化变得很重要；其优点在于，它提供了一种简单的方法来提供一些关于可以预期的奖励水平的先验知识，比如你可以将所有的动作值初始化为0。这里我们让所有的 $Q$ 估计为+5，表示积极的reward反馈，这样的初始化能够鼓励agent进行新动作的尝试，也就是探索学习。根据我们的action-value方法，一开始被选中的action的reward在新一轮优化中会比初始值要小，所以agent会在新的state上选择新的动作进行尝试。结果就是在 $Q$ 函数收敛之前，所有的动作都会被多次尝试，这也就能够在一定程度上保证所有的 $Q$ 估计最终能接近其真实的值。

下面这张图将此方法和普通的 $\epsilon$-greedy 方法作了对比，可以看到，虽然在前期我们的optimistic-initial方法表现得比 $\epsilon$-greedy 方法要差，这是因为前期我们的optimistic-initial方法尝试探索的次数比较多，而到了后期，其表现明显好于后者，因为到了后期，该方法的探索次数会越来越少。

<img src="/assets/images/sb_ch2_optimistic_value.png" width="40%"/>

这种方式在学习场景固定的情况下是一种有效的方式，但在场景变化的情况下就不行了，这很显然。

## Upper-Confidence-Bound Action Selection

探索学习方式在强化学习中属于很重要的一种方式，如果能根据其潜在优化能力并考虑估计值和最大值的近似程度，以及这些估计的不确定性来进行非贪婪式地选择动作，那么就很好。比如一个有效的挑选方式：

$$A_t=\arg\max_a[Q_t(a)+c\sqrt{ {\log t} \over {N_t(a)} }]$$

其中 $N_t(a)$ 表示动作a在t时间内被选中的次数，且 $c \gt 0$ 控制着探索学习的深度。如果 $N_t(a)=0$，那么这个动作就被视为已经满足最大值条件，即：$A_t=a$。

我们可以这样来理解这个式子：$N_t(a)$ 随着动作被选取的次数增加，导致上限置信度（UCB）越来越小。另一方面，在分子中出现不确定性估计值 $\log t$ 随着时间的推移，使得UCB越来越大，但增长越来越小。随着时间的流逝，所有动作最终都将被选中，对于具有较低价值估计或已经被选择更多次的动作，其等待时间越长，因此被选择频率越低。

UCB在10-armed testbed 上的结果如图所示，UCB经常会表现得比较良好，但随着 $\epsilon$-greedy 参数的修改，可能会比其表现得要差，而且越来越难以超越更一般的强化学习方法。 其中一个问题在于处理非平稳问题上需要有比之前提出的方法更复杂的东西。

<img src="/assets/images/sb_ch2_UCB.png" width="40%"/>

## Gradient Bandit Algorithms

我们考虑假设对于某个动作具有一定的偏爱度 $H_t(a)$，偏爱度越高，说明选择这个动作的可能性就越大，但偏爱度从reward角度来看，没有什么联系，只有一个动作相对于另一个动作的相对偏好很重要；如果我们对所有偏好度都添加1000，则对于根据soft-max分布（即吉布斯或玻尔兹曼分布）确定的动作概率没有影响，如下：

$$Pr\{A_t=a\}=\frac{ e^{H_t(a)} } { \sum_{b=1}^ke^{H_t(b)} }=\pi_t(a)$$

偏好度基于随机梯度上升的更新规则如下：

<center>$H_{t+1}(A_t)=H_t(A_t)+\alpha(R_t-\overline{R_t})(1-\pi_t(A_t))$ and $H_{t+1}(a)=H_t(a)-\alpha(R_t-\overline{R_t})\pi_t(a),\forall a \neq A_t $</center>

其中 $\overline{R_t} \in R$ 表示 $t$ 时间内平均reward，如果当前的reward水平小于平均基准线，那么显然其偏好度会下降。再来看看有平均值baseline和无平均值baseline的区别

<img src="/assets/images/sb_ch2_baseline.png" width="40%"/>

<img src="/assets/images/sb_ch2_vs_1.png" width="90%"/>

<img src="/assets/images/sb_ch2_vs_2.png" width="90%"/>

<img src="/assets/images/sb_ch2_vs_3.png" width="90%"/>

## Associative Search (Contextual Bandits)

As an example, suppose there are several different k-armed bandit tasks, andthat on each step you confront one of these chosen at random. Thus, the bandittask changes randomly from step to step. This would appear to you as a single,nonstationary k-armed bandit task whose true action values change randomly fromstep to step. You could try using one of the methods described in this chapter thatcan handle nonstationarity, but unless the true action values change slowly, thesemethods will not work very well. Now suppose, however, that when a bandit task isselected for you, you are given some distinctive clue about its identity (but not itsaction values). Maybe you are facing an actual slot machine that changes the colorof its display as it changes its action values. Now you can learn a policy associatingeach task, signaled by the color you see, with the best action to take when facing thattask—for instance, if red, select arm 1; if green, select arm 2. With the right policyyou can usually do much better than you could in the absence of any informationdistinguishing one bandit task from another.

This is an example of an associative search task, so called because it involves bothtrial-and-error learning in the form of search for the best actions and association ofthese actions with the situations in which they are best. Associative search tasksare intermediate between the k-armed bandit problem and the full reinforcementlearning problem. They are like the full reinforcement learning problem in that theyinvolve learning a policy, but like our version of the k-armed bandit problem in thateach action affects only the immediate reward. If actions are allowed to affect thenext situation as well as the reward, then we have the full reinforcement learningproblem. We present this problem in the next chapter and consider its ramificationsthroughout the rest of the book.
