# AutoGrad

import numpy as np
import matplotlib.pyplot as plt


def plot_model(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def plot_loss(loss):
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def plot_grad(grad):
    plt.plot(grad)
    plt.xlabel('Epoch')
    plt.ylabel('Gradient')
    plt.show()


class AutoGrad:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def forward(self, x, y):
        self.x = x
        self.y = y
        return self.x * self.y

    def backward(self):
        return self.y, self.x

    def __call__(self, x, y):
        return self.forward(x, y)


if __name__ == "__main__":
    x = np.random.rand(200)
    y = np.random.rand(200)
    z = x * y
    model = AutoGrad(x, y)
    plot_model(x, y, z)
    loss = []
    grad = []
    for i in range(100):
        z = model(x, y)
        loss.append(np.mean(z))
        dzdx, dzdy = model.backward()
        grad.append(np.mean(dzdx))
        x -= 0.01 * dzdx
        y -= 0.01 * dzdy
    plot_loss(loss)
    plot_grad(grad)
