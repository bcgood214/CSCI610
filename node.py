import math, random

class Node:
    def __init__(self, num_of_weights, bias, activation="logistic"):
        self.activation = activation
        self.net_input = None
        self.weights = [random.random() for _ in range(num_of_weights)]
        self.bias = bias
        self.output = None
    
    def compute_input(self, inputs):
        sum = 0
        if len(inputs) != len(self.weights):
            return "Misalignment detected"
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]
        
        self.net_input = sum + self.bias
    
    def linear(self):
        return self.net_input
    
    def linear_deriv(self):
        return 1
    
    def logistic(self):
        return 1/(1+math.exp(-1 * self.net_input))
    
    def logistic_deriv(self):
        return self.output * (1 - self.output)
    
    def activate(self):
        if self.activation == "logistic":
            self.output = self.logistic()
        if self.activation == "linear":
            self.output = self.linear()
    
    def activation_deriv(self):
        if self.activation == "logistic":
            self.deriv = self.logistic_deriv()
        if self.activation == "linear":
            self.deriv = self.linear_deriv()
    
    def compute_weights(self, delta, inputs, lr):
        self.update_weights = []
        for i in range(len(self.weights)):
            w = delta * inputs[i]
            self.update_weights.append(self.weights[i] - (lr * w))
    
    def compute_weights_and_update(self, delta, inputs, lr):
        self.update_weights = []
        for i in range(len(self.weights)):
            w = delta * inputs[i]
            self.weights[i] -= lr * w