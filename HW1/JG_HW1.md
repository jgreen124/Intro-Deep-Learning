# Introduction to Deep Learning - Homework 1

## Problem 1: K-Nearest Neighbors (KNN)

This first problem asks us to use a KNN to recognize handwritten digits.

### Dataset Information
The MNIST dataset can be downloaded using the `download_mnist.py` script. This script generates an `mnist.pkl` file that contains the data converted to numpy arrays. The MNIST dataset includes:

- `x_train`: A 60,000x784 numpy array containing flattened training images.
- `y_train`: A 1x60,000 numpy array that corresponds to the true label of the corresponding training images.
- `x_test`: A 10,000x784 numpy array where each row contains flattened versions of test images.
- `y_test`: A 1x10,000 numpy array where each component is the true label of the corresponding test images.

### Source Code for `download_mnist.py`
```python
import numpy as np
from urllib import request
import gzip
import pickle

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        request.urlretrieve(base_url + name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

if __name__ == '__main__':
    init()
```

### Source Code for `knn.py`
```python
import math
import numpy as np  
from download_mnist import load
import operator  
import time

x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000, 28, 28)
x_test  = x_test.reshape(10000, 28, 28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)

def kNNClassify(newInput, dataSet, labels, k):
    result = []
    for test_sample in newInput:
        distances = []
        for train_image in dataSet:
            distance = np.sqrt(np.sum((test_sample - train_image) ** 2)) # L2 Distance
            distances.append(distance)
        
        k_neighbors = np.argsort(distances)[:k]
        k_labels = labels[k_neighbors]
        predicted_label = np.bincount(k_labels).argmax()
        result.append(predicted_label)
    return result

start_time = time.time()
outputlabels = kNNClassify(x_test[0:20], x_train, y_train, 7)
result = y_test[0:20] - outputlabels
result = (1 - np.count_nonzero(result) / len(outputlabels))
print("---classification accuracy for knn on mnist: %s ---" % result)
print("---execution time: %s seconds ---" % (time.time() - start_time))
```

The `knn.py` script imports the `mnist.pkl` data, looping through a specified data set performing distance calculations. Then, the k-nearest neighbors are calculated to classify the image.

### `knn_tester.py` Script
```python
insert code for knn_tester here
```

The `knn_tester.py` script is a modified version of `knn.py` that runs the knn with both L1 and L2 distances as well as with different k values. 

### Output from `knn_tester.py`
```
insert results for knn_tester here
```

### Observations on L1 and L2 Distances and Accuracy as k Changes
```
insert observations about L1, L2 distances and accuracy as k changes here
```

---

## Problem 2: Linear Classifier

The second problem asks us to train a linear classifier to recognize handwritten digits.

The `linear_classifier.py` script implements a linear classifier to accomplish handwriting recognition. This linear classifier uses "Cross Entropy" for the loss function and employs "Random Search" to find the parameters W. Afterwards, the accuracy of the linear classifier is tested using the MNIST testing set.

### Source Code for `linear_classifier.py`
```python
insert code for linear_classifier.py here
```

### Results for `linear_classifier.py`
```
insert results for linear_classifier.py here
```

