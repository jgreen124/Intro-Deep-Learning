import math
import numpy as np  
from download_mnist import load
import operator  
import time

x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)

def kNNClassify(newInput, dataSet, labels, k, distance_metric='L2'): 
    result=[]
    for test_sample in newInput:
        distances = []

        for train_image in dataSet:
            if distance_metric == 'L2':
                distance = np.sqrt(np.sum((test_sample - train_image) ** 2)) #L2 Distance
            else:
                distance = np.sum(np.abs(test_sample - train_image)) #L1 Distance
            distances.append(distance)
        
        k_neighbors = np.argsort(distances)[:k]
        k_labels = labels[k_neighbors]
        predicted_label = np.bincount(k_labels).argmax()
        result.append(predicted_label)
    return result

# Test both L1 and L2 distances
for distance_type in ['L1', 'L2']:
    print(f"\nTesting {distance_type} Distance:")
    print("=" * 50)
    
    # Loop through different k values
    for k in range(1, 11):
        start_time = time.time()
        outputlabels = kNNClassify(x_test[0:1000], x_train, y_train, k, distance_type)
        result = y_test[0:1000] - outputlabels
        accuracy = (1 - np.count_nonzero(result)/len(outputlabels))
        execution_time = time.time() - start_time
        
        print(f"k={k}:")
        print(f"---classification accuracy for knn on mnist: {accuracy:.4f} ---")
        print(f"---execution time: {execution_time:.2f} seconds ---")
        print("-" * 50)
