import numpy as np
from download_mnist import load
from sklearn.preprocessing import OneHotEncoder
import time

# Load MNIST dataset
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000, -1) / 255.0
x_test = x_test.reshape(10000, -1) / 255.0

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

# Hyperparameters
num_classes = 10
num_features = x_train.shape[1]
learning_rate = 0.1
num_epochs = 100
batch_size = 128

# Initialize weights
W = np.random.randn(num_features, num_classes) / np.sqrt(num_features)
start_time = time.time()

# Training loop
for epoch in range(num_epochs):
    # Shuffle training data
    indices = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train_onehot[indices]
    
    # Mini-batch training
    for i in range(0, len(x_train), batch_size):
        batch_x = x_train_shuffled[i:i + batch_size]
        batch_y = y_train_shuffled[i:i + batch_size]
        
        # Forward pass
        scores = batch_x @ W
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Backward pass
        dscores = probs - batch_y
        dW = batch_x.T @ dscores
        
        # Update weights
        W -= learning_rate * dW / batch_size

    # Evaluate on test set
    if (epoch + 1) % 10 == 0:
        test_scores = x_test @ W
        test_preds = np.argmax(test_scores, axis=1)
        accuracy = np.mean(test_preds == y_test)
        print(f'Epoch {epoch + 1}, Test accuracy: {accuracy:.4f}')

print("---execution time: %s seconds ---" % (time.time() - start_time))

