#Introduction to Deep Learning - Homework 1

##Problem 1 (KNN)
This first problem asks us to use a KNN to recognize handwritten digits.

The MNIST dataset can be downloaded using the 'download_mnist.py' script. This script generates an 'mnist.pkl' file that contains the data converted to numpy arrays. These files are:
- 'x_train', a 60,000x784 numpy array containing flattened training images
- 'y_train', a 1x60,000 numpy array that corresponds to the true label of the corresponding training images
- 'x_test', a 10,000x784 numpy array that each row contains flattened versions of test images
- 'y_test', a 1x10,000 numpy array that each component is the true label of the corresponding test images

