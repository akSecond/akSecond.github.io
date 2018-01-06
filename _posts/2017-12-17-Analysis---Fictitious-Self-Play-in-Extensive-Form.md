---
layout: post
title: Analysis - Fictitious Self-Play in Extensive-Form Games
tags: [paper, reinforcement-learning, multi-agent]
date: 2017-12-17 00:00:00 +08:00
---



这是Deep Mind于2015年发表在JMLR上的一篇文章，文章提出了fictitious play的两种变体方法，使得fictious play能够在大型问题中得到有效的应用。这两种方法在文中分别被称为Full-width extensive-form play（XFP）和Fictitious self-play（FSP）。其实现都是基于行为策略，使用这种策略的一个好处是，可以极大的减少需要参考的动作空间和状态数量，从而减轻模拟过程或者算法对计算资源的消耗，学习速率当然也会有提升。Deep Mind的文章一般涉及的知识面都很广，交叉性很强，有必要预先了解一些博弈论的知识（虽然文章中也有数理上的说明），有必要可以参看[WikiPedia](https://zh.wikipedia.org/wiki/博弈论#范式博弈（Normal_form_game）)。

（以下所有“玩家”和“agent”意义相同）

**Fictitious Play**

fictitious play是Brown在1951年提出的一种游戏理论模型，简单来讲是一个多次重复模拟实验模型，每次模拟都会让每个玩家选择一个针对其对手的行为而言做出的最佳策略。而玩家的平均策略在某些类别的游戏中（比如二人零和博弈）均会收敛于纳什均衡（Nash Equlibrium，NE）。Leslie 和 Collins 在 2006 年给出了泛化弱化self-play模型。这种模型和通常的self-play有着类似的收敛保证，但是允许有近似best-response和扰动的平均策略更新，所以也让这个扩展模型对机器学习尤其合适。泛化弱化fictitious play是一个混合策略模型，其每次策略迭代使用凸优化线性规则，同时考虑上一次混合行为策略和次最优best-response（之后会解释）。

**Extensive-Form Equilibrium & Normal-Form Equilibrium**

> 展开式博弈和常态博弈的一个基本区别是后者的定义避免了如何计算策略的问题，游戏是怎么进行的在常态博弈下没法给出，而前者是讨论游戏进程的一个比较方便的表示形式。

*extensive-form quilibrium*

展开式博弈（Extensive-Form Euilibrium）是一种涉及到多个agents的序列化交互模型，这种基于游戏树的形式可以用 $(\mathcal{N},\mathcal{A},\mathcal{S},\mathcal{U},R)$ 这样一个元组来描述。其中 $\mathcal{N}$ 表示agents集合，$\mathcal{A}$ 表示动作空间，$\mathcal{A_i}$ 表示对应agent的专有动作空间，$\mathcal{S}$ 表示状态空间。在这棵游戏树上，每个节点代表一个状态 $s \in \mathcal{S}$，状态与其后继节点直接的连接（树的边）代表一个动作空间的子集 $\mathcal{A(s)}$。假设博弈过程是具有完美回忆（perfect-recall）属性的，那么使用 $\mathcal{U_i}$ 来表示每个agent的游戏记忆，也就是 $\mathcal{U_i}=u^i_1,a^i_1,u^i_2,a^i_2,…,u^i_n$。$R$ 表示一个pay-off映射矩阵，也就是RL中的reward。

所有agent在这场博弈中的目的都是最大化自己的收益（pay-off），而在不完美博弈中，agent只知道自己的信息状态而不知道其他agents的信息状态，实际情况比如Poker游戏便是这样：每个玩家只知道自己的手牌信息而不知道其他玩家的手牌信息，如果玩家能够记住他们自己的历史出牌情况的话（完美回忆）每次决策都会基于其动作的概率分布来进行下一次的游戏行为。

玩家的行为策略可以理解为基于其信息状态 $\mathcal{U_i}$ 的动作概率分布 $\pi^i(u) \in \Delta(\mathcal{A(u)})$，其中 $u \in \mathcal{U_i}$。用 $\Delta_b^i$ 表示玩家 $i$ 的行为策略集合，用 $\pi=(\pi_1,\pi_2,…,\pi_n)$ 表示所有玩家的策略组合，$\pi^{-i}$ 表示除去玩家 $i$ 以外的策略组合，那么玩家 $i$ 的针对所有对手策略 $\pi^{-i}$ 的best-response（最佳响应策略）为：

$$b^i(\pi^{-i})= \arg \max_{\pi^i \in \Delta^i_b}R^i(\pi^i, \pi^{-i})$$

在近似情况下，假设pay-off和最佳情况相差 $\epsilon$，那么用 $\epsilon$-NE 来表示次最优解。

*normal-form equilibrium*

常态博弈在博弈论中也称为正则博弈或者范式博弈、策略型赛局或标准赛局。常态博弈的策略（pure strategies）和展开式博弈的策略集合关系可以表示为 $\Delta^i_p \in \Delta^i_b$。之所以叫pure strategies，因其规定了玩家可能遇到的所有情况下的确定性行为，也就是每个状态下所做的动作都是唯一确定的，而另一种 mixed strategies 可以理解为基于所有 pure strategies 的概率分布下的混合纯策略。

**Realization-equivalence**

实现等价性原则在文章中被提到用来描述在扩展形式下的行为策略和常态形式下的混合策略之间的等价关系：

首先定义如果两个策略拥有相同的概率分布，那么它们就是实现等价的，同时Kuhn在1953年的理论研究证明，在具有完美记忆情况下，任何混合策略都是和行为策略实现等价的。这为之后的实现简化提供了理论依据。

**Extensive-Form Fictitious Play & Fictitious Self-play**

展开式虚拟对局（XFP）实现了best-response和当前策略的凸优化组合，从而实现在线性空间和限行时间复杂度的情况下实现策略更新。假设对于某个玩家而言，其当前的游戏状态是 $s$，那么可以得到一个关于状态的常量：$\lambda_1x_\pi(s) + \lambda_2x_b(s)$。其中 $\lambda_1 + \lambda_2=1$，原始基于当前信息状态概率分布的凸优化组合为：$\pi_{t+1}(s)=\lambda_1x_\pi(\sigma_s)\pi(s)+\lambda_2x_b(s)b(s)$，根据实现等价性原则，该行为策略的凸优化组合等价于常态博弈下的混合策略模型：$M=\lambda_1\Pi+\lambda_2B$，那么正则化后的策略更新的线性组合可以写成：

$$\pi_{t+1}(s)=\pi_{t}(s) + { {\lambda_2x_{b_{t} }(\sigma_s)(b_t(s)-\pi_t(s))} \over {\lambda_1x_{\pi_t}(\sigma_s) + \lambda_2x_{b_t}(\sigma_s)} }$$

其算法表示如下，每次迭代涉及两个操作，1）根据当前平均策略计算best-response；2）使用best-response去更新当前平均策略。

<img src="/assets/images/xfp.png" width="50%"/>

但这种方法的一个缺点也很明显，每次迭代都需要遍历所有的游戏状态，而广义下的虚拟对局只需要近似的best-response就可以进行策略的迭代更新。

Fictitious Self-play通过采样方式（或者说replay-buffer）降低了数据量，又通过强化学习和监督学习的方式把原有的计算best-response和平均策略的方法分别替换掉，从而实现一个对XFP的近似估计。具体的做法是把游戏数据（或者说玩家游戏经验，信息状态序列）分成两个集合存储，其中一个存储形式为 $(s_t,a_t,r_{t+1},s_{t+1})$ 作为强化学习的数据集来得到玩家的best-response；另一个形式为 $(s_t,a_t)$ 用于监督学习得到玩家的平均策略。在强化学习模型下，为了能够很好的解决对best-response的近似估计，玩家的对局经验都是从他们对手的策略组合中采样得到的。假设训练过程中我们每次得到一个次最优解 $\epsilon-NE$，第 $k$ 次的 $\epsilon$ 我们需要保证它在 $k \rightarrow \infty$ 的时候是收敛的 $\epsilon \rightarrow 0$，然而如果实际训练过程中MDP之间是不相关的话，玩家学习到的知识很难被传递下去。然而在虚拟对局模型中，MDP有个特殊的结构，玩家在第 $k$ 轮的平均策略组合是两个混合策略：上一局的平均策略和best-response的凸优化的组合。 这样就能够保证学到的知识能够被利用起来。

而使用监督学习部分通过拟合一个 $\mathcal{S} \rightarrow \pi(s,a)$ 的动作概率分布函数得到一个平均策略，最后再将这个平均策略和RL过程学习到的best-response进行凸优化线性组合的到混合策略模型，最后的实验结果显示FSP的近似结果在大规模场景中的近似效果比小规模的近似效果要好，几乎一致。算法描述如下：

<img src="/assets/images/fsp.png" width="50%"/>

**References**

1. [Fictitious Self-Play in Extensive-Form Games](http://jmlr.org/proceedings/papers/v37/heinrich15.pdf)
2. [WikiPedia](https://zh.wikipedia.org/wiki/博弈论#范式博弈（Normal_form_game）)
3. [Neural FSP](http://www.jianshu.com/p/bcbc41125c54)