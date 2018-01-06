---
layout: post
title: Some Algorithms For MARL
tag: [machine-learning, reinforcement-learning, algorithms]
date: 2017-10-27 19:15:00 +08:00
---

**Deep Repeated Update Q-Network**

It tries to address an issue in the way Q-learning estimate the value of an action. Ideally, if an agent could execute every possible action in parallel but identical environments at each time step, then information about all possible actions could be gathered in order to update every action value simultaneously. From this conjecture, RUQL proposes that an **action value must be updated inversely proportional to the probability of the action selected** given the policy that is being followed. Thus when an action with low probability is selected, the corresponding action-value is updated more than once. By contrast, if an action with high probability is selected, then the action-value may be updated only once. Algorithm 5 provides an initial way to formalize this intuition, while this algorithm may become unbounded in computation time as $\pi(s, a) \to 0$.

<img src="/assets/images/ruql.png" width="45%"/>

<img src="/assets/images/ruql2.png" width="45%"/>

The implement of alrorithm-6 states that if an action has a very high chance of being selected then $1 \over {\pi(s, a)} \to 1$ and standard Q-Learning is recovered. On the other hand when an action is rarely selected then not only the action-value is updated inversely proportional but also the new estimates carry more weight.

**Deep Loosely Coupled Q-Network**

Assumes that an agent is not capable of observing the full information content of the environment. Therefore an agent has to learn under which circumstances it has to act independently adn when in coordination with other agents or the information they provide. This alrogithm makes explicit considerations about multi-agents. 

*agent independence*

An independence degree $\epsilon^k_i \in [0,1]$ for agent $i$ in state $s^k_i$ determines the probability of an agent carrying on an action independently. The closer $\epsilon^k_i$ is to the upper bound, the more certainty there is for an agent to act based on its individual information regradless of the presence of other agents. We determining independence degree with the negative outcomes it receives, many methods you can use, such as Gaussian-like diffusion distribution.

**Decentralized Markove Decision Processes**

In multi-agent domains, an agent may not only depend on the information it has gathered about its environment. It will also be influenced by the choices of other agents. Naturally, these problems are partially observable. Decentralized Partially Observable Markov Decision Processes (Dec-POMDP) (Bernstein et al., 2000) have been developed as an extension of POMDPs to address situations where agents can exploit levels of coordination among them.

**Overoptimistic estimation in Q-Learning**

Double Q-network, the study addresses it by decoupling the selection and the evaluation of actions. In Sorokin et al.(2015), they extend DQN to LSTM networks to present areas of attention. And other work extends beyond the application of deep neural networks to Q-learning as it is the case in Lillicrap et al.(2015), where they present an algorithm that generalizes to continuous spaces using deterministic policy gradients.

**Multi-Agent Reinforcement Learning**

As we have seen, reinforcement learning provides an alternative to deal with the constantly changing environments. RL agents learn from experience by observing their environment and the effect of their actions. Nonetheless the transition from single agent RL to multi-agent RL offers a series challenges.

The reward that the agent may receive will not only depend on its interaction with a passive environment, In multi-agent environments, it is intertwined with the actions made by the others. Defining a goal becomes complex because the rewards are correlated and cannot simply be maximized independently, cause it should concern the global environment.

One of the biggest open issues in multi-agent environments is how to deal with non-stationarity. A policy is optimal and stationary when it is the best possible policy and it remains fixed over time. Due to the dependence of the reward function on the actions taken by other agents, good policies at a given point could not be so in the future. They are only good policies in relation to what the other agents have learned at the time the policy is applied. The exploration-exploitation dilemma becomes even more relevant under these settings. Information gathering is not only important initially but has to be done with certain recurrence while at the same time being careful that it does not destabilize the agent or agents when an appropriate coordination is required.

In practice, convergence in most complex multi-agent problems tends to be empirically verified. In some cases single RL algorithms such as Q-Learning have been used with no modification (Claus and Boutilier, 1998; Crites and Barto, 1998; Tan, 1993). However several extensions to a multi-agent domain have been proposed for cooperative tasks (Kapetanakis and Kudenko, 2005; Lauer and Riedmiller, 2000; Littman, 2001b), competitive tasks (Littman, 1994) as well as mixed tasks (Tesauro, 2003). There has two extensions to Q-Learning are presented. Each of them tries to address a concern or weakness of Q-Learning when dealing with multi-agent or non-stationary tasks. These two algorithms will serve as the basis of novel extensions to large state spaces.


