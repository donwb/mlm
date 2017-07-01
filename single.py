# this is a left turn after reading a blog post found:
# https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1?imm_mid=0f4065&cmp=em-prog-na-na-newsltr_20170701

from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        random.seed(1)

        # Model a single neuron, w/3 input connections and 1 output connection
        # asssign random weights to a 3 x 1 matrix, with values from -1 to 1
        # and mean of 0
        self.synaptic_weights = 2 * random.random((3, 1)) - 1
    
    # This describes the sigmoid itself
    def __sigmoid(self, x):
        return 1 / (1 * exp(-x))

    # This is the derivative of the sigmoid func
    # It's the gradient of the sigmoid curve
    # it indicates how confident we are about the existing wieght
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # this allows us to train the neural network through a process of trial and error
    # we adjust the synaptic weights each time
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for i in range(number_of_training_iterations):
            # Pass the training set through the neural network (single neuron)
            output = self.think(training_set_inputs)

            # calc the error
            error = training_set_outputs - output

            # multiply the error by the input and again by the gradient of the sigmoid curve
            # this means less confident weights are adjusted more
            # the means inputs, which are zero, do not change the weights
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # adjust the weights.
            self.synaptic_weights += adjustment

    def think(self, inputs):
        # pass the inputs through the single neuron network
        return self.__sigmoid(dot(inputs, self.synaptic_weights))
        

if __name__ == "__main__":
    # init the neural network
    neural_net = NeuralNetwork()

    print( "Random starting synaptic weights: ")
    print(neural_net.synaptic_weights)

    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # train the network using a training set
    # do it 10,000 times and make small adjustments each time
    neural_net.train(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic weights after training: ")
    print(neural_net.synaptic_weights )

    print("Considering new situation [1, 0, 0] -> ?: ")
    print(neural_net.think(array([1, 0, 0])))

# it's not exactly returning the right result

# Considering new situation [1, 0, 0] -> ?:
# [ 139.43645713], should have been [ 0.99993704] or similar, close to 1 basically
