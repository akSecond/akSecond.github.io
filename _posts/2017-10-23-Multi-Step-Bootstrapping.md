---
layout: post
title: Multi-Step Boostrapping
tags: [machine-learning, reinforcement-learning, sutton-book]
date: 2017-10-23 13:45:00 +08:00
---

**N-Step TD Prediction**

An important property of n-step returns is that their expectation is guranteed to be a better estimate of $v_{\pi}$ than $V_{t+n-1}$ is, in a worst-state sense. That is, the worst error of the expected n-step return is guaranteed to be less than or equal to $\eta^n$ times the worst error under $V{t+n-1}$: $\max_{s} \mid E_{\pi}[G_{t:t+n} \mid S_t=s]-v_{\pi}(s)\mid \le \eta^n \max_{s} \vert V_{t+n-1}(s)-v_{\pi}(s) \vert$. This is called the _error reduction property_ of n-step returns. 

**N-Step Sarsa**

How can n-step methods be used not just for prediction, but for control? And in this section shows how n-step methods can be combined with Sarsa in straightforward way to produce an on-policy TD control method, we call this n-step Sarsa, and the previous chapter we henceforth call one-step Sarsa or Sarsa(0).

**N-Step Off-policy Learning by Importance Sampling**

Suppose that we have two policies $\pi$ and $b$, the former is _greedy policy_ while the later is _$\epsilon$-greedy policy_, then we want use data from policy-b in policy-$\pi$, so we need take into account the difference between the tow policies by using their relative probability of taking the actions that were taken. For example, to make a simple off-policy version of n-step TD, the update for time $t$ (actually made at time $t + n$) can simply be weighted by $\rho_{t:t+n-1}$:

$$V_{t+n}(S_t) = V_{t+n-1}(S_t) + \alpha \rho_{t:t+n-1}[G_{t:t+n} - V_{t+n-1}(S_t)]$$

The importance sampling ratio
