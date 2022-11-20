import numpy as np
from typing import List, Union


class Tensor:
    # 可直接作为计算图节点
    def __init__(self, data: np.ndarray):
        self.data = data

        self.func = 'leaf'
        self.in_node: List[Tensor] = []
        self.out_deg = 0
        self.grad: Union[int, np.ndarray] = 0  # 记录 loss 对该数据的梯度

    @property
    def shape(self):
        return self.data.shape

    def dim(self):
        return self.shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nparams(self):
        return np.prod(self.shape, dtype=np.int32).item()

    def __add__(self, other):
        output = self.data + other.data
        output = Tensor(output)
        return output

    def __sub__(self, other):
        output = self.data - other.data
        output = Tensor(output)
        return output

    def __mul__(self, other):
        output = self.data * other.data
        output = Tensor(output)
        return output

    def init_grad_(self):
        # only for loss Tensor
        self.grad = 1

    def normal_(self):
        self.data = np.random.randn(*self.shape)

    def uniform_(self, a, b):
        self.data = np.random.uniform(a, b, size=self.shape)


def randn(*size):
    data = np.random.randn(*size)
    return Tensor(data)


def rand(*size):
    data = np.random.rand(*size)
    return Tensor(data)


def zeros(*size):
    data = np.zeros(size)
    return Tensor(data)


def zeros_like(x):
    data = np.zeros_like(x)
    return Tensor(data)


def ones(*size):
    data = np.ones(size)
    return Tensor(data)


def ones_like(x):
    data = np.ones_like(x)
    return Tensor(data)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def relu(x):
    return np.where(x > 0, x, np.zeros_like(x))


def onehot(x, c):
    return np.eye(c)[x]


def softmax(x):
    exp_x = np.exp(x)
    probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return probs
