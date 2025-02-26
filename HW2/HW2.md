# Intro to Deep Learning - Homework 2
Joshua Green

## Problem 1: Use back-propgation to calculate the gradients of 
$$
f(W,x) = ||\sigma(Wx)||^2
$$
with respect to $x$ and $W$. Here, $||\cdot||$ is the calculation of L2 loss, $W$ is a 3x3 matrix, $x$ is a 3x1 vector, and $\sigma(\cdot)$ is the ReUL function that performs element-wise operation.

### Solution:
First, lets denote:
$$
z = Wx\\
a = \sigma(z)\\
f(W,x) = ||a||^2 = a^Ta
$$
where $z$ is the linear transformation, $a$ is the activation function, and $f(W,x)$ is the L2 squared norm.

Since backpropogation uses the chain rule in reverse order, we can compute the gradients of $f(W,x)$ computing the gradients of $f(W,x)$ with respect to $a$ and $z$ first.

First, compute $\frac{\partial f}{\partial a}$:
$$
f = a^Ta = \sum_{i=1}^{3}a_i^2\frac{\partial f}{\partial a_i} = 2a_i\frac{\partial f}{\partial a_i} = 2a
$$

Next, compute $\frac{\partial a}{\partial z}$
Since $a = \sigma(z)$, we can compute the derivative of the ReLU function as:
$$
\sigma(z_i) = max(0,z_i)\frac{\partial \sigma(z_i)}{z_i}\\ = \begin{cases} 0 & z_i \leq 0\\ 1 & z_i > 0 \end{cases}\\
\begin{cases} 0 & z_i \leq 0\\ 1 & z_i > 0 \end{cases}\frac{\partial a}{\partial z} = diag(\sigma'(z))
$$

Next, compute $\frac{\partial z}{\partial W}$ and $\frac{\partial z}{\partial x}$:
$$
\frac{\partial z}{\partial x} = W\frac{\partial z_i}{W_{ij}} = x_j\\
$$

For $\frac{\partial f}{\partial x}$:
$$
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial a_i}\frac{\partial a_i}{\partial z_i}\frac{\partial z_i}{\partial W_{ij}} = 2a\cdot diag(\sigma'(z))\cdot W^T = 2W^T(\sigma'(z)\cdot a)
$$

For $\frac{\partial f}{\partial W}$:
$$
\frac{\partial f}{W_ij} = \frac{\partial f}{\partial a_i}\frac{\partial a_i}{\partial z_i}\frac{\partial z_i}{\partial W_{ij}} = 2a\cdot diag(\sigma'(z))\cdot x^T = 2x^T(\sigma'(z)\cdot a)
$$