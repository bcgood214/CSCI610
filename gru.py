import math
import numpy as np

def sigmoid(x):
    return 1/(1+math.exp(-1*x))

class GRU:
    def forward(self, x, h):
        self.r = sigmoid(np.dot(self.w_r, h) + np.dot(self.u_r, x))
        self.z = sigmoid(np.dot(self.w_z, h) + np.dot(self.u_z, x))
        self.h_hat = math.tanh(np.dot(np.multiply(self.r, h), self.w_h) + np.dot(self.u_h, x))
        self.h_t = (1 - self.z) * self.h_hat + self.z * h
        self.h_prev = h
        self.x_prev = x
        return self.h_t
    
    def backprop(self, out_grad):
        self.d0 = out_grad
        self.d1 = np.multiply(self.z, self.d0)
        self.d2 = np.multiply(self.h_prev, self.d0)
        self.d3 = np.multiply(self.h_hat, self.d0)
        self.d4 = np.multiply(self.d3, -1)
        self.d5 = self.d2 + self.d4
        self.d6 = np.multiply(self.d0, 1 - self.z)
        self.d7 = np.multiply(self.d5, np.multiply(self.z, 1-self.z))
        self.d8 = np.multiply(self.d6, 1-(self.h_hat**2))
        self.d9 = np.dot(self.d8, self.u_h)
        self.d10 = np.dot(self.d8, self.w_h)
        self.d11 = np.dot(self.d7, self.u_z)
        self.d12 = np.dot(self.d7, self.w_z)
        self.d14 = np.multiply(self.d10, self.r)
        self.d15 = np.multiply(self.d10, self.h_prev)
        self.d16 = np.multiply(self.d15, np.multiply(self.r, 1-self.r))
        self.d13 = np.dot(self.d16, self.u_r)
        self.d17 = np.dot(self.d16, self.w_r)
    
    def compute_gradient(self, out_grad):
        self.backprop(out_grad)
        self.dx = self.d9 + self.d11 + self.d13
        self.dh_prev = self.d12 + self.d14 + self.d1 + self.d17
        self.du_r = np.dot(self.d16, self.x_prev)
        self.du_z = np.dot(self.d7, self.x_prev)
        self.du_h = np.dot(self.d8, self.x_prev)
        self.dw_r = np.dot(self.d16, self.h_prev)
        self.dw_z = np.dot(self.d7, self.h_prev)
        self.dw_h = np.dot(self.d8, np.multiply(self.h_prev, self.r))



if __name__ == "__main__":
    gru = GRU()
    gru.w_r = np.array([1, 1])
    gru.u_r = np.array([1, 1])
    gru.w_z = np.array([2, 2])
    gru.u_z = np.array([2, 2])
    gru.w_h = np.array([1, 1])
    gru.u_h = np.array([1, 1])
    x = np.array([2, 2])
    h = np.array([3, 3])
    gru.forward(x, h)
    gru.backprop(np.array([1, 1]))
    gru.compute_gradient(np.array([1, 1]))