# I. Training Examples and ML Model

- The process involves using training examples with an ML Model $f(x_{\text{Train}}, W)$.
- The model's weights (parameters) are represented by \( W \).

## II. Softmax Classifier

- The Softmax classifier builds upon a linear classifier.
- It maps the score of class $k$ to the probability of being in that class.
- The probabilities of being in different classes sum up to 1.
- The formula to get probabilities:
  $
  P(y = k | x) = \frac{e^{s_k}}{\sum_{j} e^{s_j}}
  $
- Numeric stability must be considered for exponential terms in practice.

## III. Loss Function

- The loss function tells how good the current classifier is.
- Loss over the dataset is the average loss for all examples.
- Formula:
  $
  L = \frac{1}{N} \sum_{i=1}^{N} L_i
  $
  - Where $N$ is the number of examples.
  - $y_i$ is the correct label.
- **Mean Absolute Error (MAE) Loss**: Also called L1 loss.
  $
  L_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} | y_i - \hat{y}_i |
  $
- **Mean Square Error (MSE) Loss**: Also called L2 loss.
  $
  L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
  $
- **Cross Entropy Loss**: Negative log likelihood of the correct class as the loss.
  $
  L = - \sum_{i=1}^{N} y_i \log \hat{y}_i
  $
  - Where $y_i$ is the correct label.
  - Indicator function: $1\{x\}$ is 1 if true, 0 otherwise.

## IV. Regularization

- Different weights $W$ can have the same loss.
- Regularization helps to express preference and avoid overfitting.
- Formula:
  $
  L_{\text{total}} = L_{\text{data}} + \lambda L_{\text{reg}}
  $
  - Data loss: Model prediction should match training data.
  - Regularization: Prevent model from doing too well on training data.
  - $\lambda$: Regularization strength.
- **Overfitting**: Model fits not only the regular relation between inputs and outputs but also the sampling errors.
- **Weight Regularization**: Helps to select simple models.
- **Common Regularization**:
  - L2 regularization: $\sum W^2$
  - L1 regularization: $\sum |W|$
  - Elastic Net: Combination of L1 + L2.
  - More complex: Dropout, Batch normalization, Stochastic depth, etc.

## V. How to Find the Best Weights (W)

- The loss function quantifies the quality of any particular set of weights $W$.
- The goal of optimization is to find $W$ that minimizes the loss function.

## VI. Strategies to Minimize Loss (L)

- **Strategy #1: Random Search (very bad)**: Try many different random weights and keep track of what works best.
- **Strategy #2: Random Local Search (still bad)**: The starting position is random, and the update is local with a random update value.
- **Strategy #3: Following the Gradient (good)**: Best direction is along the steepest descent (via gradient).

## VII. Gradient Descent

- **Update via the Opposite Direction of the Gradient**
- In 1-dimension, the gradient is the derivative of a function:
  $
  \frac{d}{dx} f(x)
  $
- In multiple dimensions, the gradient is the vector of partial derivatives:
  $
  \nabla f = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right]
  $
- **Use Gradient to Update Weight**:
  $
  W = W - \epsilon \nabla L
  $
  - $\epsilon$ is a small step size (hyperparameter).
- **Proper Step Size is Important**:
  - Too small: Long training time.
  - Too big: Skips the optimal point, making it hard to converge.

## VIII. Computing Gradient

- **Numerical Method**: Small increment to calculate partial derivative.
  - **Two-sided Numerical Method**: More accurate approximation to the tangent line than the one-sided estimation for small values of $h$.
    $
    \frac{\partial f}{\partial x} \approx \frac{f(x + h) - f(x - h)}{2h}
    $
- **Mini-batch Stochastic Gradient Descent (SGD)**:
  - Vanilla gradient descent needs all training data, resulting in high memory and computation.
  - Mini-batch method splits data into batches, updating the weight for each batch.
  - Shuffling data before each epoch.
  - Good balance between SGD and batch GD: Not too fast learning, not too much memory.
  - Introduces noise, which helps to avoid local minima.
  - **Batch size** is an important hyperparameter.
- **Analytic Gradient**:
  - Loss function $L$ is a function of $W$.
  - Gradient can be calculated via calculus based on **Backpropagation**.
  - Use calculus to compute.
- **Numerical vs Analytic Gradient**:
  - Numerical gradient: Approximate, slow, easy to write.
  - Analytic gradient: Exact, fast, error-prone.
  - **Gradient check**: Always use analytic gradient but check implementation with a numerical gradient.
