---
layout: post
title: Overview - Multi-Agent
tags: [multi-agent]
date: 2017-12-14 15:35:00 +08:00
---

**1. Multi-Agent reinforcement learning algorithms**

- keep tracking of the other agents’ policy for adaptation. (opponent modeling)
- a fusion of temporal-different RL, game theory, and more general direct policy search techniques

**2. Fully cooperative tasks**

- in the absence of additional coordination mechanisms, different agents may break these ties among multiple optimal joint actions in different ways,  and the resulting joint action may be suboptimal
- the Team Q-learning algorithm avoids the coordination problem by assuming that the optimal joint actions are unique (which is not always the case)
- Distributed Q-learning algorithm and Frequency Maximum Q-value both work under deterministic tasks or static tasks

**3. A general way to solving the coordination problem is to make sure that any ties are broken by all the agents in the same way**

- Social conventions encode a priori preferences toward certain joint actions, and help break ties during action selection.
- communication can be used to negotiate action choices, partial or complete q-tables, state-measurements, rewards, learning parameters, etc.

**what’s the meaning of breaking tie ?**

*break ties among multiple optimal joint actions in different ways, the tie means joint actions array*

**4. Mixed tasks: no constraints are imposed on the reward functions of the agents, this model is most for self-interested (自利) (but not necessarily competing) agents**

- there still has some difficulties for the dynamic behavior of the agents (nonstationary). This is why most of the methods in this category focus on adaptation to the other agents

- **single-agent RL algorithms** like Q-learning can be directly applied to the multi-agent case for that these algorithms learning without being aware of (意识到) the other agents, so the nonstationary of the MARL problem invalidates most of the single-agent RL theoretical guarantees, and also for their simplicity. But it appears that for certain parameter settings, Q-learning is able to converge to a coordinated equilibrium in particular games, in other cases, unfortunately, Q-learners exhibit non-stationary cyclic behavior.

- **the agent-independent methods** has the similar structure like fully-competitive task, but there has a difference in the solver

- in current method, solver(i) returns agent i’s part of some type of equilibrium (a strategy), and eval(i) gives the agent’s expected return given this equilibrium, *the goal is to converge to an equilibrium in every state.*
	- the update requires all agents use the same algorithm for measure all actions and rewards, but it only guaranteed to maintain identical results for all the agents when solve returns consistent equilibrium strategies for all the agents
	- **so there will have a selection problem arises when the solution of solve is not unique**

- in the agent-independent methods, there has some external mechanism (coordination or negotiation) to guarantee the convergence of NE selection, such as **correlated equilibrium Q-learning & asymmetric Q-learning**

- **agent-tracking methods** estimate the policies of models (consider static or dynamic), and this category requires a *best-response* rather convergence to stationary strategies, and each agent is assumed capable to observe the other agents’ actions
	- under the static tasks, there has some important algorithms: MetaStrategy, Hyper-Q
	- under the dynamic tasks, there has some important algorithms: Non-Stationary Converging Policies

- **agent-aware (感知) methods target convergence, as well as adaptation to the other agents**
	- under the static tasks, there has some important algorithms: AWESOME
	- under the dynamic tasks, there has some important algorithms: Win-or-Learn-Fast Policy Hill-Climbing (WoLF-PHC)

**5. Application domains: distributed control, multi-robot teams, trading agents, and resource management**

- distributed control is a meta-application for cooperative multi-agent systems, agents are controller, and their environment is the controlled process
- providing domain knowledge to the agents can greatly help them to learn solutions of realistic tasks, and domain knowledge can be supplied in several forms: informative reward functions, rewarding promising behaviors rather than just the achievement of the goal
- so far, game-theory-based analysis has only been applied to the learning dynamics of the agents, while the dynamics of the environment have not been explicitly considered.

**6. Related work & Extensive overview of MARL**

- rather than estimating value functions and using them to derive policies, it is also possible to directly explore the space of agent behaviors using, e.g., nonlinear optimization techniques.
- MARL aims to provide an array of algorithms that enable multiple agents to learn the solution of difficult tasks, using limited or no prior knowledge.
