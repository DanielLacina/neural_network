import numpy as np
from tensorflow.keras.datasets import mnist



def relu(x: np.ndarray):
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray):
    return np.where(x > 0, 1, 0)

def softmax(x: np.ndarray):
    stable_x = x - np.max(x, axis=1, keepdims=True)
    e_x = np.exp(stable_x)
    sum_ex = np.sum(e_x, axis=1, keepdims=True)
    return e_x / sum_ex 

class Layer:
    def __init__(self, input_size: int, output_size: int, activation: str = "relu"):
        self.weights = np.random.rand(input_size, output_size) - .5
        self.biases = np.random.rand(1, output_size) - .5
        self.activation = activation 

    def activation_fn(self, x):
        if self.activation == "softmax":
            return softmax(x)
        else: 
            return relu(x)

    def activation_derivative(self, x):
        return relu_derivative(x)

class NeuralNetwork:
    def __init__(self, layers: list[Layer], learning_rate: float):
        self.layers = layers
        self.learning_rate = learning_rate

    def run(self, X: np.ndarray):
        _, activated_outputs = self._forward(X)
        output = activated_outputs[-1]
        return output

    def _forward(self, X):
        unactivated_outputs = []
        activated_outputs = [X]
        for layer in self.layers: 
            output = activated_outputs[-1] @ layer.weights + layer.biases 
            unactivated_outputs.append(output)
            activated_output = layer.activation_fn(output)
            activated_outputs.append(activated_output)
        return unactivated_outputs, activated_outputs

         
    def train(self, X: np.ndarray, target: np.ndarray, epoches: int, batch_size: int):
        num_samples = X.shape[0]

        for i in range(epoches):
            permutation = np.random.permutation(num_samples)
            X_shuffled = X[permutation]
            target_shuffled = target[permutation]

            for j in range(0, num_samples, batch_size):
                X_batch = X_shuffled[j:j+batch_size]
                target_batch = target_shuffled[j:j+batch_size]
                
                unactivated_outputs, activated_outputs = self._forward(X_batch)
                self._backpropagate(unactivated_outputs, activated_outputs, target_batch)
            
            _, full_activated_outputs = self._forward(X)
            predictions = np.argmax(full_activated_outputs[-1], axis=1)
            actual = np.argmax(target, axis=1)
            accuracy = np.mean(predictions == actual)
            print(f"Epoch {i}, Accuracy: {accuracy:.4f}") 

    def _backpropagate(self, unactivated_outputs, activated_outputs, target):
        dL_dY = activated_outputs[-1] - target
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            layer_outputs = unactivated_outputs[i]
            """
                this starts by indexing the second to last layer due  
                to the max index of the weights list being one less    
                than the max index of the activated outputs list 
            """
            layer_inputs = activated_outputs[i]
            if layer.activation == "softmax":
                dL_dZ = dL_dY
            else:
                dL_dZ = dL_dY * layer.activation_derivative(layer_outputs) 
            dL_dX = dL_dZ @ layer.weights.T
            dL_dW = layer_inputs.T @ dL_dZ
            layer.weights -= self.learning_rate * dL_dW
            dl_dB = np.sum(dL_dZ, axis=0, keepdims=True) 
            layer.biases -= self.learning_rate * dl_dB
            dL_dY = dL_dX


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
batch_size, input_size = 60000, 784

X = train_images.reshape((batch_size, input_size))/255
target_matrix = np.zeros((60000, 10))
target_matrix[np.arange(60000), train_labels] = 1

neural_network = NeuralNetwork(layers=[Layer(input_size=784, output_size=100, activation="relu"), Layer(input_size=100, output_size=10, activation="softmax")], learning_rate=.01)
neural_network.train(target=target_matrix, epoches=1, X=X,batch_size=64)
X = test_images.reshape((10000, input_size))/255
target_matrix = np.zeros((10000, 10))
target_matrix[np.arange(10000), test_labels] = 1
outputs = neural_network.run(X)



      






