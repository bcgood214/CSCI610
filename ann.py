import math, random
from node import Node

class ANN:
    def __init__(self, num_of_inputs, layers, default_activation="logistic"):
        self.costfunc = "mse"
        self.lr = 0.5
        self.layers = []
        for i in range(len(layers)):
            if i == 0:
                layer = [Node(num_of_inputs, bias=random.random(), activation=default_activation) for _ in range(layers[i])]
            else:
                layer = [Node(layers[i-1], bias=random.random(), activation=default_activation) for _ in range(layers[i])]

            self.layers.append(layer)
    
    def forward_pass(self, inputs):
        values = inputs
        for layer in self.layers:
            outputs = []
            for neuron in layer:
                neuron.compute_input(values)
                neuron.activate()
                outputs.append(neuron.output)
            values = outputs
    
    def layer_activation(self, layer_ind, actfunc):
        layer_len = len(self.layers[layer_ind])
        for i in range(layer_len):
            self.layers[layer_ind][i].activation = actfunc
    
    def mse(self, label):
        sum = 0
        for i in range(len(label)):
            sum += 1/2 * (label[i] - self.layers[-1][i].output)**2
        
        return sum
    
    def mse_deriv(self, label, label_ind):
        # print(label_ind)
        return (label[label_ind] - self.layers[-1][label_ind].output) * -1
    
    def cost(self, inputs, label):
        self.forward_pass(inputs)
        if self.costfunc == "mse":
            return self.mse(label)
    
    def cost_deriv(self, label, label_ind):
        if self.costfunc == "mse":
            return self.mse_deriv(label, label_ind)
        
    def compute_output_delta(self, label):
        output_delta = []
        for i in range(len(label)):
            output_deriv = self.cost_deriv(label, i)
            self.layers[-1][i].activation_deriv()
            output_delta.append(output_deriv * self.layers[-1][i].deriv)
        
        return output_delta

    
    def compute_delta(self, label):
        # print(label)
        self.delta = []
        output_delta = self.compute_output_delta(label)
        self.delta.append(output_delta)

        for i in range(len(self.layers) - 2, -1, -1):
            delta_prime = [0 for _ in self.layers[i]]
            for j in range(len(self.layers[i+1])):
                for w in range(len(self.layers[i+1][j].weights)):
                    delta_prime[w] += self.delta[-1][j] * self.layers[i+1][j].weights[w]
            
            for j in range(len(self.layers[i])):
                self.layers[i][j].activation_deriv()
                delta_prime[j] = self.layers[i][j].deriv * delta_prime[j]
            
            self.delta.append(delta_prime)
    
    def backward_pass(self, input, label):
        self.compute_delta(label)

        for i in range(len(self.layers) - 1, -1, -1):
            delta_ind = (len(self.layers) - 1) - i
            if i > 0:
                inputs = [self.layers[i-1][j].output for j in range(len(self.layers[i-1]))]
            else:
                inputs = input
            
            for j in range(len(self.layers[i])):
                self.layers[i][j].compute_weights_and_update(self.delta[delta_ind][j], inputs, self.lr)
    
    def train(self, inputs, labels, epochs=100):
        for _ in range(epochs):
            for i in range(len(inputs)):
                self.forward_pass(inputs[i])
                self.backward_pass(inputs[i], labels[i])



if __name__ == "__main__":
    nn = ANN(2, [2, 2])
    nn.layers[0][0].bias = 0.35
    nn.layers[0][0].weights[0] = 0.15
    nn.layers[0][0].weights[1] = 0.20
    nn.layers[0][1].weights[0] = 0.25
    nn.layers[0][1].weights[1] = 0.30
    nn.layers[0][1].bias = 0.35
    nn.layers[1][0].bias = 0.6
    nn.layers[1][0].weights[0] = 0.40
    nn.layers[1][0].weights[1] = 0.45
    nn.layers[1][1].weights[0] = 0.50
    nn.layers[1][1].weights[1] = 0.55
    nn.layers[1][1].bias = 0.6

    cost = nn.cost([0.05, 0.1], [0.01, 0.99])
    print(cost)
    inputs = [ [0.05, 0.1] ]
    labels = [ [0.01, 0.99] ]
    nn.train(inputs, labels, epochs=1)
    cost = nn.cost([0.05, 0.1], [0.01, 0.99])
    print(cost)