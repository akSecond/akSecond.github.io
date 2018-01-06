---
layout: post
title: LSTM Tutorial
tags: [machine-learning, lstm]
date: 2017-10-02 16:41:00 +08:00
---

> the LSTM network includes a set of **recurrently connected subnets**, which we call as "*memory block*", these memory block are made with one or more self-conncted memory cells and three multiplicative units—the input, output, and forget gates—that provide continuous analogues of write, read and reset operations for the cells.

**_what is the sensitivity in NN cell?_**

The sensitivity of our NN cell has a connection with the impact of information (or input data) to current NN cell, if the sensitivity of a NN cell is low, we gonna say that current NN cell do not care about current input data.

**_why RNNs have the vanishing gradient problem?_**

We have some ways to make a statement for this problem.  In the aspect of transfering, the sensitivity of our network will decay over time as new inputs overwrite the activations of the hidden layer and the network 'forgets' the former inputs. In the image below, the shading indicates the sensitivity of unfolded network, lighter and more vanish.

<img src="/assets/images/unfold_lstm.png" width="40%">

(I'll make a explanation in the aspect of mathematics in the future)

**_what is LSTM memory block look like?_**

The memory block include some gates and activation function which seted for controlling the gates, in the image below states it.

<img src="/assets/images/memory_block_lstm.png" width="40%">

The tiny black balls represents 3 gates for collecting activation from inside and outside the "_memory block_" with non-linear activation, and theres has some tings we need focus on:

- No activation function is applied within the cell
- $\int_f$ is usually the logistic function, so that it can control the state of gate (open := 1 and close := 0) 
- $\int_s \& \int_h$ are usually tanh or logistic function, though in some cases ‘h’ is the identity function
- The weighted ‘peephole’ connections from the cell to the gates are shown with dashed lines. All other connections within the block are unweighted (or equivalently, have a fixed weight of 1.0). 	

**_How LSTM make preservation of gradient information?_**

Suppose the shading of the nodes indicates their sensitivity to the input at time one, so that black maximum while white are entirely insensitive, and the state of input, forget and output gates are displayed below, left and above the hidden layer.

<img src="/assets/images/transfer_state.png" width="40%">

<img src="/assets/images/three_state.png" width="40%">

For simplicity, all gates are either entirely open (‘O’) or closed (‘—’). The memory cell ‘remembers’ the first input as long as the forget gate is open and the input gate is closed. The sensitivity of the output layer can be switched on and off by the output gate without affecting the cell.

**_Maybe LSTM is not the best answer_**

The above discussion raises an important point about the influence of preprocessing. If we can find  a way to transform a task containing long range contextual dependencies into one containing only short-range dependencies before presenting it to a sequence learning algorithm, then architectures such as LSTM become somewhat redundant. Suppose we have a raw speech sample whose frequency is 40kHz, it is obviously that this speech sample is a _long range contextual_ sample. If we can find a way to transform this sample to 100Hz series of mel-frequency cepstral coefficients, it becomes feasibel to model the data with hidden Markov model. Nonetheless, if such a transform is difficult or unknown, or if we simply
wish to get a good result without having to design task-specific preprocessing methods, algorithms capable of handling long time dependencies are essential.
