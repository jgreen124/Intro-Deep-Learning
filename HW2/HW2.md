# Intro to Deep Learning - Homework 2
Joshua Green

## Problem 1: Use back-propagation to calculate the gradients of $f(W,x) = ||\sigma(Wx)||^2$ with respect to $x$ and $W$.
Here, $||\cdot||$ is the calculation of L2 loss, $W$ is a 3x3 matrix, $x$ is a 3x1 vector, and $\sigma(\cdot)$ is the ReUL function that performs element-wise operation.

### Solution:
First, let's multiply the 3x3 matrix $W$ by the 3x1 matrix $x$, and let's call the resultant 3x1 matrix $z$. From this, it follows that:
$$
z = \begin {bmatrix}
W_{1,1}x_1+W_{1,2}x_2+W_{1,3}x_3 \\
W_{2,1}x_1+W_{2,2}x_2+W_{2,3}x_3 \\
W_{3,1}x_1+W_{3,2}x_2+W_{3,3}x_3 
\end {bmatrix}
$$

Let $a = \sigma(Wx) = \sigma(z)$. Therefore,
$$
a = \begin {bmatrix}
max(0, z_1) \\ max(0, z_2) \\ max(0,z_3) \end {bmatrix} = \begin {bmatrix}  a_1 \\ a_2 \\ a_3 \end {bmatrix} \\
$$
$$
a = \begin {cases} z_i & z_i>0 \\ 0 & z_i<0 \end {cases}
$$

Therefore, $f(W,x) = ||a||^2$

Now, we can find $\nabla _W f$ and $\nabla _x f$ by working backwards with chain rule.

First, find $\nabla _a f$
$$
\frac{\partial f}{\partial a_i} = \frac{\partial}{\partial a}(a_1^2 + a_2^2 + a_3^2) = 2a_i = \begin {bmatrix} 2a_1 \\ 2a_2 \\
 2a_3\end {bmatrix}
$$
$$
\nabla _a f = 2a
$$

Next, find $\nabla _z f$
$$
\nabla _z f = \frac{\partial f}{\partial z} = \frac{\partial f}{\partial a}\frac{\partial a}{\partial z}
$$

The derivative of the ReLU function is:
$$
\frac{\partial a}{\partial z} = \begin {cases} 1 & z_i > 0 \\ 0 & z_i < 0 \end {cases} = \begin {bmatrix} I_{(z_1>0)} & 0 & 0 \\
0 & I_{(z_2>0)} & 0 \\ 0 & 0 & I_{(z_3>0)} \end {bmatrix}
$$

As a result,
$$
\nabla _z f = \frac{\partial f}{\partial z} = \begin {bmatrix} 2a_1 \cdot I_{(z_1>0)} \\ 2a_2 \cdot I_{(z_2>0)} \\ 2a_3 \cdot I_{(z_3>0)}\end {bmatrix}
$$
$$
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial z}\frac{\partial z}{\partial x}
$$
$$
\frac{\partial z}{\partial x} = W_{k,i}
$$

Therefore,
$$
\nabla _x f = \frac{\partial f}{\partial x} = \sum_j{\frac{\partial f}{\partial z_j}\frac{\partial z_j}{\partial x}} = \sum_j{2z_i \cdot I_{(z_i>0)}W_{j,i}} = W^T \cdot \nabla _z f
$$

and
$$
\nabla _w f = \nabla _z f \cdot x^T
$$