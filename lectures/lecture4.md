# I. Backpropagation Overview

- Backpropagation is used to calculate $\frac{\partial L}{\partial W}$.
- It is related to training examples, ML models, loss functions, and updating weights $W$.
- It uses a computational graph.

## II. Key Components

- **ML Model:** $f(x, W)$
- **Loss Function:** Includes data loss and regularization loss.
  - Examples: L1, L2, Cross Entropy (CE)
- **Regularization:** $R(W)$
- **Weights:** Model weights are parameters $W$ that need updating.

## III. Chain Rule

- The chain rule is essential for calculating gradients.
- For a function $q(x)$, the chain rule is:
  $$
  \frac{\partial L}{\partial x} = \frac{\partial L}{\partial q} \cdot \frac{\partial q}{\partial x}
  $$

## IV. Gradients

- Gradients include an upstream gradient and a local gradient.
- **Upstream Gradient:** The gradient of the loss function with respect to the output of a given layer or operation.
- **Local Gradient:** The gradient of the output of a layer or operation with respect to its inputs.
- Formula:
  $$
  \frac{\partial L}{\partial x} = \text{Upstream gradient} \times \text{Local gradient}
  $$

## V. Examples

- The lecture notes provide several examples of calculating gradients using the chain rule.
- Example:
  - Given $z = x + y$, if $x = -2$ and $y = 5$, then $z = 3$.
  - $\frac{\partial z}{\partial x} = 1, \frac{\partial z}{\partial y} = 1$.
- Another example includes the function:
  $$
  f(w,x) = e^{\frac{w_0 x_0}{w_1 x_1 w_2}}
  $$

## VI. ADD and MUL Operations

- **ADD** acts as a distributor.
- **MUL** acts as a switcher.

## VII. Sigmoid Function

- The lecture notes mention the sigmoid function and sigmoid gate.
- The derivative of the sigmoid function is:
  $$
  \sigma'(x) = \sigma(x) (1 - \sigma(x))
  $$

## VIII. Vectorization

- When $x, y, z$ are vectors, gradients will be in the format of the Jacobian matrix.
- The lecture notes refer to matrix calculus.

## IX. Updating Weights

- Weights are updated using the formula:
  $$
  w_i = w_i - \text{step size} \times \frac{\partial L}{\partial w_i}
  $$

## X. Additional Topics

- Softmax classifier
- Numerical methods for computing gradients
- Regularization (L1, L2)
- Cross Entropy (CE) loss
