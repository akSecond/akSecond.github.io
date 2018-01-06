---
layout: post
title: Prove of Double Q-learning
tags: [machine-learning, reinforcement-learning]
date: 2017-11-14 00:20:00 +08:00
---

In some stochastic environments the well-known reinforcement learning algorithm Q-learning performs very poorly. This poor performance is caused by large overestimations of action values. These overestimations result from a positive bias that is introduced because Q-learning uses the maximum action value as an approximation for the maximum expected action value. The update of Q-learning is:

$$Q_{t+1}(s_t,a_t)=Q_t(s_t,a_t)+\alpha_t(s_t,a_t)(r_t+\gamma \max_a Q_t(s_{t+1},a)-Q_t(s_t,a_t))$$

While this update method has a issue—overestimation, cause' this formula always choose the maximum value of $Q_t(s_{t+1},a \mid a \in \mathcal{A}(s_{t+1}))$. 

**Estimating the maximum expected value**

Assume that there has $M$ random variables 

$$X=\{ X_1, X_2,…,X_M \}$$

and the maximum expected value of thise valriable set is:

$$\max_{i}E[X_i]$$

Cause we have no knowledge of the function form and parameters of the underlying distribution of the variables in $X$, so we need construct appoximations for $E[X_i]$ for all $i$. Assume we have a sample set $S=\bigcup^M_{i=1}S_i$, in which $S_i$ is a sample set of $X_i$. Also we assume that each $S_i$ obey the iid condition (独立同分布), and unbias estimation can be the average of each $S_i$, then we have:

$$E[X_i]=E[\mu_i]\thickapprox\mu_i(S) \overset{def}{=}{ {1\over \vert S_i \vert}{\sum_{s\in S_i}s} }$$

Which $\mu_i$ represents the estimator of $X_i$, so, if we wanna get the $\max_i{E[X_i] }$, we can calculate the $\max \mu_i(S)$ to get the approximation, while $E\{ \max_i \mu_i \}$ is the unbias of $\max_i \mu_i$.

We introduce two conception, PDF (probability density function) $f_i(x)$ and CDF (cumulative distribution function) $F_i(x)$:

$$F_i(x)=\int_{-\infty}^{x}f_i(x)dx$$

and $F_i(x) = \int_{-\infty}^{\infty}f_i(x)dx=1$, so, the CDF-formula of $\max_i{E[X_i] }$ is: $\max_i\int_{-\infty}^{\infty}f_i(x)dx$.

**Double Estimator**

Here, we assume there has two independent sample set $S_A$ and $S_B$, while $S_A=\bigcup_{i=1}^M{S_A^i}$ and $S_B=\bigcup_{i=1}^M{S_B^i}$, and $S_A^i \bigcap S_B^i=\emptyset$. As we mentioned above, we create estimatior $\mu_i^A$ or $\mu_i^B$ for each subset of $S_A$ or $S_B$.

There has some differences as we mentioned above, I believe you have detected them. Now, we will use one sample set to select the maximal estimates in one estimator set, suppose we select $S_A$ ans $\mu_A(S)$ to do that, then we define the maximal estimates:

$$Max^A(S) \overset{def}{=} \{j \mid \mu_j^A(S)\}$$

If there has many $j$, then we select one from them randomly. And because $\mu^B$ is an independent, unbiased set of estimators, we have $E[\mu_j^B]=E[X_j]$ for all $j$, including all $j \in Max^A$. Let $a^{*}$ be the extimator that maximizes $\mu^A$: $\mu_{a^\star}^A(S)\overset{def}{=}\max_i\mu_{i}^A(S)$. Then we can use $\mu_{a^\star }^B$ as an estimator for $\max_iE[X_i]$, and we have:

$$\max_iE[X_i]=max_iE[\mu_i^B]\thickapprox \mu_{a^\star}^B$$

Now, assume that the underlying PDFs are continuous, then the probability $P(j=a^{\star})$ equals to the probability that all $i \ne j$ give lower estimates. Thus $\mu_j^A(S)=x$ is maximal for some value $x$ with probability $\Pi_{i \ne j}^M{P(\mu_i^A<x)}$, then $P(j=a^\star)$ is:

$$P(\mu_j^A=x)\Pi_{i\ne j}^M{P(\mu_i^A<x)} \overset{def}{=} \int_{-\infty}^{\infty}f_j^A(x)\Pi_{i\ne j}^M{F_i^A(x)dx}$$

And $E{\mu_{a^{*} }^B}$ is a unbiased estimation of $\mu_{a^\star}^B$, the expectation of $\mu_{a^\star }^B$ is:

$$\sum_{j}^M{P(j=a^{\star})E[\mu_j^B] }=\sum_{j}^M{E[\mu_j^B]\int_{-\infty}^{\infty}f_{j}^A(x)\Pi_{i\ne j}^M{F_i^A(x)dx} }$$

**Is unbias or lower than the unbias ?**

As we metioned above, we have 

$$E[X_i]=E[\mu_i^A]=E[\mu_i^B]$$

and let

$$\mathcal{M}\overset{def}{=}\{j \mid E[X_j]=\max_iE[X_i]\}$$

be the set of elements that maximize the expectated values, so the expectated value can be written as：

$$\begin{align}E[\mu_{a^\star}^B]  &=P(a^{\star} \in \mathcal{M})E[\mu_{a^{\star} }^B \mid a^{\star} \in \mathcal{M}]+P(a^{\star} \notin \mathcal{M})E[\mu_{a^{\star} }^B \mid a^{\star} \notin \mathcal{M}] \\ &=P(a^{\star} \in \mathcal{M})\max_{i}E[X_i]+P(a^{\star} \notin \mathcal{M})E[\mu_{a^{\star} }^B \mid a^{\star} \notin \mathcal{M}] \\ &\le P(a^{\star} \in \mathcal{M})E[\mu_{a^{\star} }^B \mid a^{\star} \in \mathcal{M}]+P(a^{\star} \notin \mathcal{M})\max_i E[X_i] \\ &=\max_i{E[X_i] }\end{align}$$

Where the inequality is strict if and only if $P(a^{\star} \notin \mathcal{M}) > 0$. This happens when the variables have different expected values, but their distributions overlap. In contrast with the single estimator, the double estimator is unbiased when the variables are iid, since then all expected values are equal and $P(a^{\star} \in \mathcal{M})=1$.

**Prove convergence of Double Q-learning**

Although we solve the estimation issue, there has another big issue needs to be solved, that's the convergence of Double Q-learning. Cause' its convergenc e is inherited from Q-learning, so I will show the prove of convergence of Q-leraning (*also I acquiesce that you know the Double Q-learning algorithm and its update process*).

All we know that the update process of Q-learning is: 

$$Q(S_{t+1}, a_{t+1}) := Q(S_{t+1}, a_{t+1}) + \alpha(r+\gamma\max_{a}Q(S_{t}, a) - Q(S_{t+1}, a_{t+1}))$$

Now, let us define some ABC of Markov decision process $(\mathcal{X}, \mathcal{A}, \mathcal{P}, r)$, where

- $\mathcal{X}$ is the finite state-space
- $\mathcal{A}$ is the finite action-space
- $\mathcal{P}$ represents the transition probabilities
- $r$ represents the reward function

Then, we have the function $r$ defined as:

$$r: \mathcal{X} \times \mathcal{A} \times \mathcal{X} \rightarrow \mathbb{R}$$

And the value of a state $x$ is defined for a sequence of controls $\{A_t\}$, as:

$$J(x, \{A_t\})=\mathbb{E}\Bigg[\sum_{t=0}^{\infty}\gamma^tR(X_t,A_t) \mid X_0=x\Bigg]$$

The optimal value functions is defined, for each $x \in \mathcal{X}$ as:

$$V^{\star}(x)=\max_{\mathcal{A} }J(x,\{A_t\})$$

and verifies

$$V^{\star}(x)=\max_{a\in \mathcal{A}}\sum_{y\in \mathcal{X} }\mathbb{P}_a(x,y)[r(x,a,y)+\gamma V^{\star}(y)]$$

$x$ and $y$ represent two states which $x$ switch to $y$ through action $a$, and we call $(x, a, y)$ as a transition of Markov decision process $(\mathcal{X}, \mathcal{A}, \mathcal{P}, r)$.

And from above, we define the optimal Q-function, $Q^{\star}$ as:

$$Q^{\star}(x, a)=\sum_{y\in \mathcal{X} }[r(x,a,y)+\gamma V^{\star}(y)]$$

We can define a contraction operator $\mathbb{H}$ which defined for a generic function: $q: \mathcal{X} \times \mathcal{A} \rightarrow \mathbb{R}$ as:

$$(\mathbb{H}(x,a)=\sum_{y\in \mathcal{X} }\mathbb{P}_a(x,y)[r(x,a,y)+\gamma \max_{b\in \mathcal{A} }q(y,b)])$$

And the meaning of construction operator can be intrepreted as follow:

$$\Vert \mathbb{H}q_1 - \mathbb{H}q_2 \Vert_{\infty} \le \Vert q_1 - q2\Vert_{\infty}$$

and we can prove it:

$$\begin{align}\Vert \mathbb{H}q_1 - \mathbb{H}q_2 \Vert_{\infty} &= \max\Bigg\vert \sum_{y\in\mathcal{X} }[r(x,a,y)+\gamma \max_{b\in \mathcal{A} }q_1(y, b) - r(x,a,y) + \gamma\max_{b\in \mathcal{A} }q_2(y,b)]\Bigg\vert \\ &=\max_{x,a}\gamma \Bigg\vert \sum_{y\in \mathcal{X} }\mathbb{P}_a(x,y)[\max_{b\in \mathcal{A} }q_1(y,b)-\max_{b\in \mathcal{A} }q_2(y,b)] \Bigg\vert  \\  &\le\max_{x,a}\gamma \sum_{y\in \mathcal{X} }\mathbb{P}_a(x,y)\Bigg\vert \max_{b\in\mathcal{A} }q_1(y,b)-\max_{b\in\mathcal{A} }q_2(y,b) \Bigg\vert  \\  &\le \max_{x,a}\gamma \sum_{y\in \mathcal{X} }\mathbb{P}_a(x,y)\max_{z,b}\vert q_1(z,b)-q_2(z,b) \vert \\  &=\max_{x,a} \gamma \sum_{y\in \mathcal{X} }\mathbb{P}_a(x,y)\Vert q_1-q_2 \Vert_{\infty}  \\  &=\gamma\Vert q_1 - q_2\Vert_{\infty} \end{align}$$

Back to the first formula at this block, as we see, if we wann $Q$-function converge, then we need promise that $\alpha(r+\gamma\max_{a}Q(S_{t}, a) - Q(S_{t+1}, a_{t+1}))$ should converge. So, it the key of convergence.

Let $F_t(x,a)=r(x,a,y)+\gamma\max_{b\in\mathcal{A} }Q_t(y, b) - Q^{\star}(x, a)$, so we have:

$$\begin{align}\mathbb{E}[F_t(x,a) \mid \mathcal{F}_t] &=\sum_{y\in \mathcal{X} }\mathbb{P}_a(x,y)r(x,a,y)+\gamma\max_{b\in\mathcal{A} }Q_t(y, b) - Q^{\star}(x, a)] \\ &= (\mathbb{H}Q_t)(x,a)-Q^{\star}(x,a)\end{align}$$

as we mentioned before, the operator $\mathbb{H}$ si a constraction operator, so we can rewrite the second item $Q^{\star}$ as $\mathbb{H}Q^{\star}$, and then we have:

$$\mathbb{E}[F_t(x,a) \mid \mathcal{F}_t]=(\mathbb{H}Q_t)(x,a)-(\mathbb{H}Q^{\star})(x,a)$$

If we use the prove process above, we can get a conclusion easily:

$$\Big\Vert\mathbb{E}[F_t(x,a) \mid \mathcal{F}_t]\Big\Vert=\Big\Vert(\mathbb{H}Q_t)(x,a)-(\mathbb{H}Q^{\star})(x,a)\Big\Vert_{\infty} \le \gamma\Vert Q_t-Q^{\star} \Vert_{\infty}$$

So far, we have proved the convergence of Q-learning, also Double Q-learning.

**References**

1. [Double Q-learning](https://papers.nips.cc/paper/3964-double-q-learning)
2. [Convergence of Q-learning](http://users.isr.ist.utl.pt/~mtjspaan/readingGroup/ProofQlearning.pdf)

*if you have some questions, please contact me, especially the approach e-mail*