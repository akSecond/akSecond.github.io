---
layout: post
title: Monte Carlo Methods
tags: [machine-learning, reinforcement-learning, sutton-book]
date: 2017-10-27 12:10:00 +08:00
---

## Off-policy Prediction via Importance Sampling

*To be continued...*

## Off-policy Monte Carlo Control

In off-policy method, estimating the value of a policy and controling are separated, it is the mainly difference between on-policy and off-policy. In off-policy, we separate all possible policies into _behavior policy_ and _target policy_. The later is inclued in the former, while it is the real policy that is evaluated and improved. For gurantee this rule which _target policy_ should has the chance to visit all _behavior policy_, we need make a promise that the probability of all _behavior policy_ should larger than zero. We define _behavior policy_ as policy $b$, and _target policy_ as policy $\pi$, then the policy $b$ may be $\epsilon$-soft greedy policy.

<img src="/assets/images/off_policy_monte_carlo.png" width="40%"/>

A potential problem is that this method learns only from the tails of episodes, when all of the remaining actions in the episode are greedy. If nongreedy actions are common, then learning will be slow, particularly for states appearing in the early portions of long episodes. Potentially, this could greatly slow learning. There has been insufficient experience with off-policy Monte Carlo methods to assess how serious this problem is. If it is serious, the most important way to address it is probably by incorporating temporal-difference learning, the algorithmic idea developed in the next chapter. Alternatively, if γ is less than 1, then the idea developed in the next section may also help significantly.

## Discounting-Aware Importance Sampling

If we consider discounting reward in a long episode, then there maybe some problem. Suppose discounting index $\gamma=0$, and the length of episode is 100, then if we  use importance sampling as the old form:

$$\rho={ {\pi(A_0 \mid S_0)\pi(A_1 \mid S_1)…\pi(A_{99} \mid S_{99})} \over {b(A_0 \mid S_0)b(A_1 \mid S_1)…b(A_{99} \mid S_{99})} }$$

While the return from time 0 will then be $G_0=R_1$. In ordinary importance sampling, the return will be scaled by the entire product, but it is really only necessaryto scale by the first factor, by $\rho={ {\pi(A_0 \mid S_0)} \over {b(A_0 \mid S_0)} }$. The other 99 factors are irrelevant. They do not change the expected update, but they add **enormously to its variance**. In some cases they could even make the variance infinite. So, how we can avoid this bad condition?

**degree of partial termination**

The essence of the idea is to think of discounting as *determining a probability of termination* or, equivalently, a degree of partial termination. That is if we terminate at the first step, then return $G_0$; if terminate after two steps, then to the degree of $\gamma(1-\gamma)$, producing a return of $R_1 + \gamma R2$. The degree of termination on the third step is thus $(1-\gamma)\gamma^2$, with the $\gamma^2$ reflecting that termination did not occur on either of the first two steps. The partial returns here are called flat partial returns:

$$\overline{G}=R_{t+1} + R_{t+2}+…+R_{h}, 0 \le t \lt h \le T$$

The conventional full return Gt can be viewed as a sum of flat partial returns as suggested above as
follows:

<img src="/assets/images/degree_formula.png" width="50%"/>

Then we define the new ordinary importance-sampling and weighted important-sampling as follow:

<img src="/assets/images/new_ordinary_formula.png" width="50%"/>

<img src="/assets/images/new_weighted_formula.png" width="50%"/>

## Per-Reward Importance Sampling

As I writed in the former text, in importance sampling, if we concered about only the reward at time $t$, then the $\pi_{t+k} \over b_{t+k}$ after time $t$ are irrelevant. And all the other ratios are independent random variables whose expected value is one:

$$E_{A_k \thicksim b} \lgroup { {\pi(A_k | S_k)} \over {b(A_k | S_k)} } \rgroup=\sum_a{b(a|S_k){ {\pi(a|S_k)} \over {b(a|S_k)} } }=\sum_a{\pi(a|S_k)}=1$$

so, we can make a conclusion: $E[\rho_{t:T-1}R_{t+1}]=E[\rho_{t:t}R_{t+1}]$. Braodcast to all $R_k$, then the formula of $G_t$ is:

$$\rho_{t:t}R_{t+1}+\gamma\rho_{t:t+1}R_{t+2}+\gamma^2\rho_{t:t+2}R_{t+3}+…+\gamma^{T-t-1}\rho_{t:T-1}R_T$$

It is also a unbias estimator as the ordinary importance sampling, we named it as *pre-reward importance sampling*. Is there a per-reward version of weighted importance sampling? This is less clear. So far, all the estimators that have been proposed for this that we know of are not consistent (that is, they do not converge to the true value with infinite data).


​			
​		
​	
