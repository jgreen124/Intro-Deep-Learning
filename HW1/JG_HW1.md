# Introduction to Deep Learning - Homework 1

## Problem 1 (KNN)
This first problem asks us to use a KNN to recognize handwritten digits.

The MNIST dataset can be downloaded using the `download_mnist.py` script. This script generates an `mnist.pkl` file that contains the data converted to numpy arrays. The MNIST dataset includes: 
- `x_train` a 60,000x784 numpy array containing flattened training images
- `y_train` a 1x60,000 numpy array that corresponds to the true label of the corresponding training images
- `x_test` a 10,000x784 numpy array that each row contains flattened versions of test images
- `y_test` a 1x10,000 numpy array that each component is the true label of the corresponding test images

The source code for `download_mnist.py` is shown below:
```
import numpy as np
from urllib import request
import gzip
import pickle

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
#    base_url = "http://yann.lecun.com/exdb/mnist/"
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()
#    print ((load()[0]).shape)
def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

if __name__ == '__main__':
    init()
```