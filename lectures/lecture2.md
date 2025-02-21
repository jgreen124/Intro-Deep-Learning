# I. What is Machine Learning?

- **Definition**: A machine learning (ML) algorithm is an algorithm that can learn from data. A computer program is said to learn if its performance (P) on a task (T) improves with experience (E).
- **Task (T)**: What the ML algorithm is used for. Examples include classification, translation, and denoising. In this course, image classification in computer vision is frequently used as an example.
- **Performance (P)**: A quantitative metric to evaluate the algorithm. Different tasks have different performance measures.
- **Experience (E)**: The effect of the ML algorithm on a dataset.

---

# II. Types of Machine Learning Based on Experience

- **Supervised Learning**: Learns from labeled examples, where each example is a pair consisting of an input object and a desired output value (a.k.a. supervisory signal).
- **Unsupervised Learning**: Draws inferences from datasets consisting of input data without labeled responses and is used to find hidden patterns or groupings in data.
- **Reinforcement Learning**: Learns by interacting with an environment, creating a feedback loop between the ML system and the environment.

---

# III. Image Classification Example

- **Computer vision** is a successful field for Deep Neural Networks (DNNs), and image classification is a core task.
- **Images in computers**: Each pixel is represented with a number, and the computer views an image as a big grid of numbers.
- **Challenges**:
  - **Semantic Gap**: Computer vision differs from human vision.
  - **Variability**: Different sizes, shapes, and angles of objects in images.
- **Data-Driven Approach**:
  1. Collect a dataset of images and labels.
  2. Use Machine Learning to train a classifier.
  3. Evaluate the classifier on new images.

---

# IV. A Naive Approach: Finding Similar Images

- **Memorization**: Memorize all data and labels and predict the label of the most similar training images.
- **Nearest Neighbors**: Find the nearest neighbors in the training images for a test image.
- **Distance Metric**:
  - L-distance is a good metric.
  - **L1 (Manhattan) distance**: $d(I_1, I_2) = \sum |I_1 - I_2|
$
  - **L2 (Euclidean) distance**: $d(I_1, I_2) = \sqrt{\sum (I_1 - I_2)^2} $
- **Python Code Example**:
  - Memorize training data.
  - For each test image, find the nearest train image and predict the label of the nearest image.
- **Time Complexity**: With $N$ training examples, training is $O(1)$ (quick), but prediction is $O(N)$ (slow).

---

# V. K-Nearest Neighbors (KNN)

- **Improvement**: Take a majority vote from $K$ closest points.
- **Hyperparameters**:
  - What is the best value of $K$ to use for voting?
  - What is the best distance to use for measuring?
  - Hyperparameters are not adapted by the machine learning algorithm itself.
  - In DNN, examples of hyperparameters include the number of layers/neurons, learning rate, and weight decay.

---

# VI. Train, Validation, and Test Datasets

- **Training Set**: Data used to discover predictive relationships and help learn the model.
- **Test Set**: Data used to assess the strength and utility of a predictive relationship and help evaluate a learned model.
- **Validation Set**: Examples used to tune the hyperparameters.
- **Workflow**: Train Model → Tune Model → Evaluate Model.
- **Important Rule**: Never use test data to train the model!
- **Cross-validation**:
  - When the dataset is small, split the data and try each fold as validation, and average.
  - 5-fold cross-validation can be used for KNN.
  - Not too frequent in deep learning.

---

# VII. Summary of Non-parametric KNN Training Examples

- **ML Model**: $f(x_{train})$
- **ML Model**: $f(x_{test}, x_{train})$
- **Pros**:
  - Fast Training
  - Easy to implement
- **Cons**:
  - Slow Testing
  - Need to store all training data

---

# VIII. Parametric Machine Learning Model Training

- **ML Model**: $f(x_{train}, W)$
- **Loss Function**: Used to update $W$ (model weights/parameters).
- **ML Model**: $f(x_{test}, W)$
- **Pros**:
  - Slow Training
  - Fast Testing
  - Does not store training data

---

# IX. Linear Classifier Training Examples

- **ML Model**: $f(x_{train}, W)$
- **Loss Function**: Used to update $W$ (model weights/parameters).
- **ML Model**: $f(x_{test}, W)$
- **Input**: 32x32x3 (RGB channel) image flattened to a length-3072 vector $x$.
- **Components**:
  - $W$ (weight)
  - $b$ (bias)
- **Score**: Used to evaluate if the model is well-trained.
- **Visual Viewpoint**: Template matching.
- **Bias**:
  - What happens if inputs are all zeros without bias?
  - **Function approximation viewpoint**: A linear classifier approximates an (assumed) linear relationship.
  - Weights and Bias can be unified.
