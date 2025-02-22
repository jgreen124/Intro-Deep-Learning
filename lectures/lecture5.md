# I. Neural Network Overview

- Multi-layer neural networks can be built on linear classifiers.
- The neural network (NN) consists of multiple layers, including an input layer, output layer, and hidden layers.
- Neurons between adjacent layers are typically connected, but neurons within the same layer are not.

## II. Neural Network Structure and Function

- The neurons in the input layer are "transparent," meaning their output is the input, which ensures consistent representation.
- Hidden layer neurons perform two main operations: accumulation of product and applying an activation function.
- Connections between neurons have weights, which are parameters that should be learned.
- Output layer neurons have special activation functions like softmax or identity functions; they can also be standard neurons if followed by another classifier.
- Activation can be considered another layer.

## III. Activation Functions

- **Sigmoid:** Squashes numbers to the range $(0,1)$ and is still used in Recurrent Neural Networks (RNNs). However, it can cause saturated neurons, which "kill" the gradients, and is not zero-centered, leading to slow learning.
- **Tanh:** Squashes numbers to the range $[-1,1]$, is zero-centered (preferred over sigmoid), and is also used in RNNs.
- **ReLU (Rectified Linear Unit):** Squashes numbers to the range $[0, \infty]$, does not saturate in half of the region, and is popular in Convolutional Neural Networks (CNNs) and Fully Convolutional Networks (FCNs). ReLU offers low-cost computation and fast convergence. However, it can "kill" gradients when $x < 0$.
- **Leaky ReLU:** Squashes numbers to the range $(-\infty, \infty)$ and does not saturate in the full region. It uses a small $\alpha$ (e.g., 0.01) that can be learned.
- **Choosing Activation Functions:** ReLU is typically chosen, but be careful with learning rates. If ReLU doesn't work well, try leaky ReLU/ELU. Tanh can sometimes be used, but sigmoid is not recommended for CNNs but can be used in RNNs.
- **Importance:** Activation functions provide non-linearity.

## IV. Vectorization

- Vectorization is always used.

## V. Representation Power and Hidden Layers

- Neural Networks are universal function approximators.
- Hidden layers are needed to solve problems like XOR, which are not linearly separable. Multi-layer perceptions draw multiple lines.

## VI. Network Depth

- Increasing depth or width can improve performance.
- Deeper networks need fewer neurons to achieve the same approximation error. Deeper architectures facilitate hierarchy learning.

## VII. Vanishing Gradient

- Solutions for vanishing gradients include better initialization, faster hardware, and using residual blocks.
