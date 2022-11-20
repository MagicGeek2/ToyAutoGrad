
class BaseOptimizer:
    def __init__(self, parameters):
        self.parameters = parameters
        self.lr = 1

    def update(self):
        for param in self.parameters:
            param.data = param.data - self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0


class SGD(BaseOptimizer):
    def __init__(self, parameters, lr=1e-3):
        super().__init__(parameters)
        self.lr = lr
