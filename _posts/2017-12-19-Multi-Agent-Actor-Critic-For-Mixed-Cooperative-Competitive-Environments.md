---
layout: post
title: Multi-Agent Actor-Critic For Mixed Cooperative-Competitive Environments
tags: [paper, multi-agent, reinforcement-learning]
date: 2017-12-19 00:00:00 +08:00
---

**Why propose this framework for Multi-Agent ?**

- Q-learning is not effective under non-stationary environment
- policy-gradient suffers from a variance that increase as the number of agents grows
- at this paper, authors proposal an adaptation of actor-critic methods that consider *action policies of other agents*

**Why the former methods are poorly suited to multi-agent environments ?**

- non-stationary cause the changes in the agents' own policies(that is not explaineable), and this issue also become a challenge of learning stability and prevents the straightforward use of past experience replay, which is crucial for stabilizing deep Q-learning
- in multi-agent system, it is common that coordination is required while policy gradient suffers from high variance

**How do we do ?**

> the authors in this paper propose a general-purpose multi-agent learning algorithm

1. learned policies only use local information at execution time
2. no explicitly communication structure: does not assume a differentiable model of the environment dynamics or any particular structure on the communication method between agents
3. not only cooperative, but also competitive or mixed task, for this setting is more natural

**Differ with centralized critic function**

- critic use some addtional information while do not use at test time, in this paper, authors use the policies of other agents


- the centralized critic function explicitly use the decision-making policies of the other agents, while the authors of this paper let agents learn *approximate models of other agents online and effictively use them in their own policy learning procedure*
- introduce a method to improve the stability of multi-agent policy by training agent with an **ensemble of policies** (**QUESTION**: use the ensemble policies to do what ? RL ?)

**How ensemble policies work ?**

- still the issue — **non-stationary**. For the non-stationary, the agents' policies always change, and under the setting of competitive, it is true that agents can derive a strong policy by overfitting to the behavior of their competitors (**WHY?**)
- for learning a more robust model, training a collection of $K$ different sub-policies will work
- training collecton of $K$ different sub-policies
  - at each episode, randomly select one particular sub-policy $\mu_i^{(k)}$ for each agent to execute
  - for agent $i$, maximizing the ensemble objective function: $J_e(\mu_i)=\mathbb{E}_{k \sim unif(1,K),s \sim p^\mu, a \sim \mu_i^{(k)} }[R_i(s,a)] $
  - **QUESTION**: how to matain the collection of sub-policy

**DPG(Deterministic Policy Gradient) Algorithms**

it is alsow posible to extend the policy gradient framework to deterministic policies $\mu_{\theta}: \mathcal{S} \rightarrow \mathcal{A}$. So we can rewrite the objective function $J$ as:

$$\Delta_{\theta}J(\theta)=\mathbb{E}_{s\sim \mathcal{D} }[\nabla_{\theta}\mu_{\theta}(a \mid s)\nabla_aQ^{\mu}(s,a) \mid_{a=\mu_{\theta}(s) }]$$

While the derive $\nabla_aQ$ implies that the action space should be continuous. *DDPG(Deep deterministic policy gradient)* is a variant of DPG where the policy $\mu$ and critic $Q^{\mu}$ are approximated with deep neural networks. DDPG is an off-policy algorithm, and samples trajectories from a replay buffer of experiences that are stored throughout training. DDPG also makes use of a target network, as in DQN

**Multi-Agent Actor Critic**

>  follow the above setting, the authors of this paper propose a simple extension of actor-critic policy gradient methods where the critic is augmented with the extra information about the policies of other agents

- suppose a game with $N$ agents with policies $\pi=\{\pi_1,\pi_2,…,\pi_n\}$ parameterized with $\theta=\{\theta_1,\theta_2,…,\theta_n\}$
- the actor learn a $Q$ function for each agent $i$ which accepts actions of all agents with some state information 

**MADDPG**

- a primary motivation behind MADDPG is that if we know the actions taken by all agents, the environment is stationary event as the policies change.
- while under the real conditions, we cannot know the actions of other agents, also their observation or policies, but we can approximate their policies from their observations.
- then we can replace the real action input of critic with approximation action: $\hat{y}=r_i+\gamma Q_i^{\mu'}{(x', \mu_i^{'1}(o1)),\mu_i^{'2}(o2),…,\mu_i^{'N}(oN))}$
- ​