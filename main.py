import numpy as np
from tensorflow.keras.datasets import mnist



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sigmoid_output = sigmoid(x)
    return sigmoid_output * (1 - sigmoid_output)


class NeuralNetwork:
    def __init__(self, dimensions: list[int], learning_rate: float):
        self.weights = [np.random.randn(dimensions[i], dimensions[i + 1]) for i in range(len(dimensions) - 1)]
        self.biases = [np.zeros((1, dim)) for dim in dimensions[1:]]
        self.learning_rate = learning_rate

    def run(self, inputs=None):
        input_ = []
        for W_, B_ in zip(self.weights, self.biases): 
            output = inputs[-1] @ W_ + B_ 
            transformed_output = sigmoid(output)
            if inputs is not None:
                inputs.append(output)
            input_ = transformed_output
        output = input_
        return output

         
    def train(self, epoches, target: np.ndarray):
        for _ in range(epoches):
            inputs = [X]
            self.run(inputs)
            self.backpropagate(inputs, target)
                            
    def backpropagate(self, inputs, target):
        outputs = inputs[-1]
        dL_dY = outputs - target
        dL_dX = dL_dY    
        for i in range(len(self.weights) - 1, -1, -1):
            layer_outputs = inputs[]
            layer_inputs = inputs[i]
            dL_dZ = dL_dX * sigmoid_derivative(layer_inputs) 
            dL_dX_minus_one = dL_dZ @ self.weights[i].T
            dL_dX = dL_dX_minus_one
            dL_dW = layer_inputs.T @ dL_dZ
            self.weights[i] -= self.learning_rate * dL_dW
            dl_dB = np.sum(dL_dZ, axis=0, keepdims=True) 
            self.biases[i] -= self.learning_rate * dl_dB

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
batch_size, input_size = 60000, 784
learning_rate = .01
epoches = 10

X = train_images.reshape((batch_size, input_size))/255
target_matrix = np.zeros((60000, 10))
target_matrix[np.arange(60000), train_labels] = 1

neural_network = NeuralNetwork(dimensions=[784, 392, 191, 10], learning_rate=.01)
neural_network.train(1, target_matrix)

      






