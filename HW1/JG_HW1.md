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
---

Next, the `mnist.pkl` file can be imported into the `knn.py` script. The source code for `knn.py` is shown below:
```
import math
import numpy as np  
from download_mnist import load
import operator  
import time
# classify using kNN  
#x_train = np.load('../x_train.npy')
#y_train = np.load('../y_train.npy')
#x_test = np.load('../x_test.npy')
#y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)
def kNNClassify(newInput, dataSet, labels, k): 
    result=[]
    ########################
    # Input your code here #
    ########################
    for test_sample in newInput:
        distances = []

        for train_image in dataSet:
            distance = np.sqrt(np.sum((test_sample - train_image) ** 2)) #L2 Distance
            #distance = np.sum(np.abso(test_sample - train_image)) #L1 Distance
            distances.append(distance)
        
        k_neighbors = np.argsort(distances)[:k]
        k_labels = labels[k_neighbors]
        predicted_label = np.bincount(k_labels).argmax()
        result.append(predicted_label)
    #####################
    # End of your code  #
    #####################
    return result

start_time = time.time()
outputlabels=kNNClassify(x_test[0:20],x_train,y_train,7)
result = y_test[0:20] - outputlabels
result = (1 - np.count_nonzero(result)/len(outputlabels))
print ("---classification accuracy for knn on mnist: %s ---" %result)
print ("---execution time: %s seconds ---" % (time.time() - start_time))
```

The `knn.py` script imports the `mnist.pkl` data, looping through a specified data set doing distance calculations. Then, the k nearest neighbors are calculated to classify the image. The `knn_tester.py` script is a modified version  of `knn.py` that loops through, testing different k values and uses both L1 and L2 distance calculations. The source code for `knn_tester.py` is shown below:
```
insert code for knn_tester here
```
The output from the `knn_tester.py` script is shown below:
```
insert results for knn_tester here
```

```
insert observations about L1, L2 distances and accuracy as k changes here
```

---
## Problem 2 (Linear Classifier)
The second problem asks us to train a linear classifier to recognize handwritten digits.

 The `linear_classifier.py` scipt implements a linear classifier to accomplish the handwriting recognition. This linear classifier uses "Cross Entropy" for the loss function and uses a "Random Search" to find the parameters W. Afterwards, the accuracy of the linear classifier is tested using the MNIST testing set. The code for `linear_classifier.py` is shown below:
 ```insert code for linear_classifier.py here``` 
```insert results for linear_classifier.py here```