import numpy as np
from tensorflow.keras.datasets import mnist


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

label = train_labels[0]

X = train_images.reshape((60000, 784))/255
W1 = np.random.randn(784, 100)   
W2 = np.random.randn(100, 10)   

output1 = X @ W1
predictions = output1 @ W2
target_matrix = np.zeros((60000, 10))
target_matrix[np.arange(60000), train_labels] = 1
# dL/dy  
error_matrix = predictions - target_matrix

#backpropagation
gradient1_w = (output1.T @ error_matrix) / 60000
gradient1_x = error_matrix @ W2.T
gradient1_b = (error_matrix.sum(axis=0)) / 60000

W2 -= 0.01 * gradient1_w

gradient2_w = (X.T @ gradient1_x) / 60000

print(gradient2_w)
W1 -= 0.01 * gradient2_w


