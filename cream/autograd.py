from cream.utils import Tensor
import numpy as np
from cream.utils import onehot


def add_node(output: Tensor, func, *inputs):
    output.func = func
    output.in_node = list(inputs)
    for input in inputs:
        input.out_deg += 1
    return output


def matmul_grad(node: Tensor):
    W, x = node.in_node  # W (n,m) ; x (..., m)
    pre_grad = node.grad  # pre_grad (..., n)
    if x.ndim == 1:
        x.grad = x.grad + np.einsum('n,nm->m', pre_grad, W.data)
        W.grad = W.grad + np.einsum('n,m->nm', pre_grad.data, x.data)
    else:
        x.grad = x.grad + np.einsum('b...n,nm->b...m', pre_grad, W.data)
        W.grad = W.grad + np.einsum('b...n,b...m->nm', pre_grad.data, x.data)


def add_grad(node: Tensor):
    x, b = node.in_node  # b (m) ; x (..., m)
    pre_grad = node.grad  # pre_grad (..., m)
    x.grad = pre_grad
    if x.ndim == 1:
        b.grad = b.grad + pre_grad
    else:
        b.grad = b.grad + np.einsum('b...m->m', pre_grad)


def mean_grad(node: Tensor):
    x, *_ = node.in_node
    pre_grad = node.grad
    x.grad = x.grad + pre_grad * np.ones_like(x.data) / x.nparams


def sigmoid_grad(node: Tensor):
    x, *_ = node.in_node
    pre_grad = node.grad
    x.grad = x.grad + pre_grad * node.data * (1 - node.data)


def relu_grad(node: Tensor):
    x, *_ = node.in_node
    pre_grad = node.grad
    x.grad = x.grad + pre_grad * \
        np.where(x.data > 0, np.ones_like(x.data), np.zeros_like(x.data))


def softmax_grad(node: Tensor):
    logits, *_ = node.in_node
    pre_grad = node.grad
    if logits.ndim == 1:
        local_grad = np.einsum('m,n->mn', node.data, -
                               node.data) + np.eye(node.shape[-1])
        logits.grad = logits.grad + np.einsum('m,mn->n', pre_grad, local_grad)
    else:
        local_grad = np.einsum('b...m,b...n->b...mn',
                               node.data, -node.data) + np.eye(node.shape[-1])
        logits.grad = logits.grad + \
            np.einsum('b...m,b...mn->b...n', pre_grad, local_grad)


def l2_loss_grad(node: Tensor):
    input, target = node.in_node
    pre_grad = node.grad
    input.grad = input.grad + pre_grad*(input.data - target.data)


def bce_loss_grad(node: Tensor):
    probs, label = node.in_node
    pre_grad = node.grad
    probs.grad = probs.grad + pre_grad * (probs.data - label.data)


def ce_loss_grad(node: Tensor):
    probs, label = node.in_node
    pre_grad = node.grad
    probs.grad = probs.grad + pre_grad * \
        (probs.data - onehot(label.data, probs.shape[-1]))


def leaf_grad(node: Tensor):
    ...


func_grad_map = dict(
    matmul=matmul_grad,
    add=add_grad,
    mean=mean_grad,
    leaf=leaf_grad,
    sigmoid=sigmoid_grad,
    relu=relu_grad,
    softmax=softmax_grad,
    l2_loss=l2_loss_grad,
    bce_loss=bce_loss_grad,
    ce_loss=ce_loss_grad,
)


def compute_grad(node: Tensor):
    assert node.func in func_grad_map
    grad_func = func_grad_map[node.func]
    grad_func(node)


def backward(loss: Tensor):
    loss.init_grad_()
    stack = [loss]
    while stack != []:
        node = stack.pop()
        node: Tensor
        # 计算对所有 child 的 grad, 乘以自己现有的 grad, 加到每个 child 的 grad 属性上
        compute_grad(node)
        for child in node.in_node:
            child.out_deg -= 1
            if child.out_deg == 0:
                stack.append(child)
