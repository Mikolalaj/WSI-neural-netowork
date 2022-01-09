
import random
import numpy as np


class Network:

    def __init__(self, sizes: list):
        '''
        sizes contains the number of neurons in the respective
        layers of the network. The biases and weights for the network
        are initialized randomly, using default_weight_initializer
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = None
        self.biases = None
        self.weight_init()

    def weight_init(self):
        '''
        This random initialization gives our stochastic gradient descent
        algorithm a place to start from. 
        Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.
        Initialize the biases using a Gaussian distribution with mean 0
        and standard deviation 1.
        '''
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]

    def feedforward(self, input):
        '''Return the output of the network'''
        for bias, weight in zip(self.biases, self.weights):
            input = sigmoid(np.dot(weight, input)+bias)
        return input

    def stochastic_gradient_descent(self, training_data, epochs, batch_size, learn_rate, lambda_, test_data):
        '''
        Train the neural network using mini-batch stochastic gradient
        descent. The training_data is a list of tuples (x, y)
        representing the training inputs and the desired outputs.
        '''
        
        training_data_len = len(training_data)

        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[j:j+batch_size] for j in range(0, training_data_len, batch_size)]
            for mini_batch in mini_batches:
                self.update_network(mini_batch, learn_rate, lambda_, len(training_data))

            accuracy = self.accuracy(test_data)
            print(f'Epoch {i+1}. Accuracy: {accuracy}%')


    def update_network(self, mini_batch, learn_rate, lambda_, n):
        '''
        Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch. The
        mini_batch is a list of tuples, lambda_ is the regularization
        parameter, and n is the total size of the training data set.
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for image, digit in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(image, digit)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-learn_rate*(lambda_/n))*w-(learn_rate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learn_rate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backpropagation(self, image, digit):
        '''
        Return a tuple (nabla_b, nabla_w) representing the
        gradient for the cost function. nabla_b and nabla_w
        are layer-by-layer lists of numpy arrays, similar
        to self.biases and self.weights
        '''
        # creating lists of zeros
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = image
        activations = [image] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation)+bias
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        # quadratic cost
        delta = (activations[-1] - digit) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[1-layer].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-1-layer].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        '''
        Return the % of correct result that the neural network predicted.

        The flag convert should be set to False if the data set is
        validation or test data, and to True if the data set is the
        training data. (Just for the optimization - it's faster like that)
        '''
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]

        result_accuracy = round(sum(int(x == y) for (x, y) in results)/len(data)*100, 2)
        return result_accuracy


def sigmoid(z):
    '''The sigmoid function.'''
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    '''Derivative of the sigmoid function.'''
    return sigmoid(z)*(1-sigmoid(z))
