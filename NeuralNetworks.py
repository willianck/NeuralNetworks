import numpy as np
from sklearn.utils import shuffle


# Neural network from scratch from watching 31bluebrowns  videos about Neural network  on youtube

# given an array representing the number size of the network
class Network:
    def __init__(self, network_size):
        self.size = network_size
        self.num_layers = len(network_size)
        # initialize random weights
        self.weights = [np.random.randn(network_size[i + 1], network_size[i]) * 1.5 for i in range(self.num_layers - 1)]
        # initialize random biases
        self.biases = [np.random.randn(network_size[i], 1) * 3 for i in range(1, self.num_layers)]

    # get the activation of the next layer of neurons from previous input neurons
    def calc_forward_activation(self, input_n):
        act = [np.zeros(s) for s in self.size]
        x = input_n.transpose()
        act[0] = input_n
        for i in range(1, self.num_layers):
            act[i] = sigmoid(np.dot(self.weights[i - 1], x) + self.biases[i - 1])
            x = act[i]
        return act

    # Calculate the stochastic gradient descent to minimise cost function
    # We use a step length of 100 but this can be varied
    def sgd(self, train_data, target, mini_batch_size, learning_rate, step_length):
        shuffled_data, shuffled_target = shuffle(train_data, target)
        input_batches = [shuffled_data[i:i + mini_batch_size] for i in range(0, len(shuffled_data), mini_batch_size)]
        output_batches = [shuffled_target[i:i + mini_batch_size] for i in
                          range(0, len(shuffled_target), mini_batch_size)]

        for step in range(step_length):
            for i in range(len(input_batches)):
                self.calc_sgd(input_batches[i], output_batches[i], learning_rate)
                self.calculate_cost(input_batches[i], output_batches[i], mini_batch_size, i)

    # updates the weights and biases for every step taken into the gradient descent
    def calc_sgd(self, data, output, learning_rate):
        for i in range(len(data)):
            x, y = self.backpropagation(data[i], output[i])
            step_size_weight = [np.multiply(x[i], learning_rate) for i in range(len(x))]
            step_size_bias = [np.multiply(y[i], learning_rate) for i in range(len(y))]
            self.weights = [np.subtract(self.weights[i], step_size_weight[i]) for i in range(len(x))]
            self.biases = [np.subtract(self.biases[i], step_size_bias[i]) for i in range(len(y))]

    # performs backpropagation on the neural network to get a vector matrix of
    # the negative gradient of the Cost function
    def backpropagation(self, inputs, output):
        pd_weights = [np.zeros(w.shape) for w in self.weights]
        pd_biases = [np.zeros(b.shape) for b in self.biases]
        act = self.calc_forward_activation(inputs)
        weighted_sum = [np.dot(self.weights[0], inputs.transpose()) + self.biases[0]]
        for i in range(1, self.num_layers - 1):
            z = np.dot(self.weights[i], act[i]) + self.biases[i]
            weighted_sum.append(z)
        dependent_derivative = sigmoid_derivative(weighted_sum[-1] * (2 * (act[-1] - output)))
        pd_weights[-1] = np.dot(dependent_derivative, act[-2].transpose())
        pd_biases[-1] = dependent_derivative
        for i in range(2, self.num_layers):
            dependent_derivative = np.dot(self.weights[-i + 1].transpose(), dependent_derivative) * \
                                   sigmoid_derivative(weighted_sum[-i])

            pd_weights[-i] = np.dot(dependent_derivative, act[-i - 1].transpose())
            pd_biases[-i] = dependent_derivative
        return pd_weights, pd_biases

    def calculate_cost(self, x, y, n, b):
        act = [self.calc_forward_activation(x[i]) for i in range(len(x))]
        outs = [y[0]]
        s = 0
        for i in range(1, len(y)):
            outs.append(y[i])

        for i in range(len(y)):
            k = act[i]
            s = s + (k[-1] - outs[i]) ** 2
        av = s / n
        print('---------------- batch simulation', b)

    def evaluate(self, test_data, target_data):
        test_results = [(np.argmax(self.calc_forward_activation(x)))
                        for x, in test_data]
        return sum(int(x == y) for (x, y) in (test_results, target_data))


# sigmoid function used to cast the output value between 0 and 1
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# sigmoid derivative function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def main():
    net = Network([50, 10, 6])
    target = np.random.choice([0, 1], size=50, p=[1. / 3, 2. / 3])
    train_data = np.random.choice([0, 1], size=(50, 50), p=[1. / 3, 2. / 3])
    net.sgd(train_data, target, 10, 0.05, 100)


main()
