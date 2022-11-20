import numpy as np
from cream.utils import Tensor
from cream.utils import zeros, zeros_like, ones_like, ones
from cream.utils import sigmoid, relu, softmax
from cream.autograd import add_node


class Module:
    def __init__(self,):
        self.requires_grad = True

    def requires_grad_(self, requires_grad):
        self.requires_grad = requires_grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def parameters(model):
    all_parameters = []
    for attr_str in dir(model):
        if '__' in attr_str:
            continue
        attr = getattr(model, attr_str)
        if isinstance(attr, Tensor):
            all_parameters.append(attr)
        elif isinstance(attr, Module):
            all_parameters.extend(parameters(attr))
    return all_parameters


class Affine(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight: Tensor = zeros(out_dim, in_dim)
        self.bias: Tensor = zeros(out_dim)

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.init_modules_()

    def init_modules_(self):
        bound = 1/np.sqrt(self.in_dim)
        self.weight.uniform_(-bound, bound)

    def _linear(self, x):
        output = np.einsum('ij,...j->...i', self.weight.data, x.data)
        output = Tensor(output)
        return output

    def forward(self, x):
        tmp = self._linear(x)
        if self.requires_grad:
            tmp = add_node(tmp, 'matmul', self.weight, x)
        output = tmp + self.bias
        if self.requires_grad:
            output = add_node(output, 'add', tmp, self.bias)
        return output


class Mean(Module):
    def forward(self, x):
        mean = np.mean(x.data)
        mean = Tensor(mean)
        if self.requires_grad:
            mean = add_node(mean, 'mean', x)
        return mean


class Sigmoid(Module):
    def forward(self, x):
        output = sigmoid(x.data)
        output = Tensor(output)
        if self.requires_grad:
            output = add_node(output, 'sigmoid', x)
        return output


class ReLU(Module):
    def forward(self, x):
        output = relu(x.data)
        output = Tensor(output)
        if self.requires_grad:
            output = add_node(output, 'relu', x)
        return output


class Softmax(Module):
    def forward(self, logits):
        probs = softmax(logits.data)
        probs = Tensor(probs)
        if self.requires_grad:
            add_node(probs, 'softmax', logits)
        return probs


class L2Loss(Module):
    def forward(self, input, target):
        l2dist = (input.data - target.data)**2
        loss = np.mean(l2dist) / 2
        loss = Tensor(loss)
        if self.requires_grad:
            loss = add_node(loss, 'l2_loss', input, target)
        return loss


class BCELoss(Module):
    def forward(self, probs, label):
        # logits: (b,1) ; label: (b,)
        loss = - \
            np.mean(
                np.log(np.where(label.data[..., None] == 1, probs.data, 1 - probs.data)))
        loss = Tensor(loss)
        if self.requires_grad:
            add_node(loss, 'bce_loss', probs, label)
        return loss


class CrossEntropyLoss(Module):
    def forward(self, probs, label):
        # logits: (b,c) ; label: (b,)
        target_probs = np.take_along_axis(
            probs.data, label.data[..., None], axis=1)  # (b,1)
        loss = -np.mean(np.log(target_probs))
        loss = Tensor(loss)
        if self.requires_grad:
            add_node(loss, 'ce_loss', probs, label)
        return loss
