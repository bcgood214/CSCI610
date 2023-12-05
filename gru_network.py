import math
from gru import GRU
import numpy as np
from numpy import random

class GRUNetwork:
    def __init__(self, cell_len, weight_len):
        self.cells = [GRU() for _ in range(cell_len)]
        self.gen_weights(weight_len)
        self.reset_gradient()
        for cell in self.cells:
            self.init_cell_weights(cell)

    def loss(self, target, prediction):
        self.target = target
        self.prediction = prediction
        return (1/2)*(target-prediction)**2
    
    def d_loss(self):
        return (self.target-self.prediction) * -1
    
    def gen_weights(self, weight_len):
        self.w_r = random.rand(weight_len)
        self.u_r = random.rand(weight_len)
        self.w_z = random.rand(weight_len)
        self.u_z = random.rand(weight_len)
        self.w_h = random.rand(weight_len)
        self.u_h = random.rand(weight_len)
    
    def init_cell_weights(self, cell):
        cell.w_r = self.w_r
        cell.u_r = self.u_r
        cell.w_z = self.w_z
        cell.u_z = self.u_z
        cell.w_h = self.w_h
        cell.u_h = self.u_h
    
    def reset_gradient(self):
        self.dw_r = 0
        self.du_r = 0
        self.dw_z = 0
        self.du_z = 0
        self.dw_h = 0
        self.du_h = 0
    
    def inc_gradient(self, dw_r, du_r, dw_z, du_z, dw_h, du_h):
        self.dw_r += dw_r
        self.du_r += du_r
        self.dw_z += dw_z
        self.du_z += du_z
        self.dw_h += dw_h
        self.du_h += du_h

    def forward(self, x, h, target):
        self.series_len = 0
        self.ht = h
        for xt in x:
            h_next = self.cells[self.series_len].forward(xt, self.ht)
            self.ht = h_next
            self.series_len += 1
        
        return self.loss(target, self.cells[self.series_len-1].h_t)
    
    def backward(self):
        out_grad = self.d_loss()
        for i in range(self.series_len-1, -1, -1):
            self.cells[i].compute_gradient(out_grad)
            self.inc_gradient(
                self.cells[i].dw_r,
                self.cells[i].du_r,
                self.cells[i].dw_z,
                self.cells[i].du_z,
                self.cells[i].dw_h,
                self.cells[i].du_h
            )
            out_grad = self.cells[i].dh_prev
    
    def update(self, lr=0.0001):
        self.w_r -= lr * self.dw_r
        self.u_r -= lr * self.du_r
        self.w_z -= lr * self.dw_z
        self.u_z -= lr * self.du_z
        self.w_h -= lr * self.dw_h
        self.u_h -= lr * self.du_h
    
    def update_cells(self):
        for cell in self.cells:
            self.init_cell_weights(cell)

if __name__ == "__main__":
    x = [np.array([0.1]), np.array([0.2]), np.array([0.3])]
    h = np.array([0])
    print(f"Input sequence: {x}")
    target = np.array([0.4])
    print(f"Target output: {target}")
    network = GRUNetwork(3, 1)
    print(f"MSE loss: {network.forward(x, h, target)}")
    print(f"Network output before training: {network.cells[2].h_t}")
    # variable for number of epochs (feel free to adjust the value)
    epochs = 1000
    for epoch in range(epochs):
        network.forward(x, h, target)
        network.backward()
        network.update(lr=0.5)
        network.update_cells()
        network.reset_gradient()
    
    print(f"MSE loss after training: {network.forward(x, h, target)}")
    print(f"Network output after training: {network.cells[2].h_t}")
