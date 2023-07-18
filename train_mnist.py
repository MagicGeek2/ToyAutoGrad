import numpy as np
from cream import Module, backward, Tensor, parameters
import cream.module as module
from cream.optim import SGD
from tqdm.auto import tqdm

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def cycle(dl):
    while True:
        for data in dl:
            yield data


def PIL_to_array(img):
    return np.array(img)


def flatten(x: np.ndarray):
    return x.flatten()


def scale_image(x):
    # uint8 -> [-1,1]
    x = x.astype(np.float32) / 255
    x = 2*x-1
    return x


def descale_image(x):
    # [-1,1] -> uint8
    x = (x+1)/2
    x = (x*255).astype(np.uint8)
    return x


class Classifier(Module):
    def __init__(self):
        super().__init__()
        self.affine1 = module.Affine(784, 128)
        self.act1 = module.ReLU()
        self.affine2 = module.Affine(128, 10)
        self.to_prob = module.Softmax()

    def forward(self, x):
        h = self.affine1(x)
        h = self.act1(h)
        h = self.affine2(h)
        # h = self.act2(h)
        # h = self.affine3(h)
        probs = self.to_prob(h)
        return probs

    def predict(self, x):
        probs = self(x)
        pred = np.argmax(probs.data, axis=-1)
        pred = Tensor(pred)
        return pred


if __name__ == '__main__':

    batch_size = 16
    num_workers = 4
    image_transforms = T.Compose([
        PIL_to_array,
        flatten,
        scale_image,
    ])
    ds = torchvision.datasets.MNIST(
        'data', train=True, transform=image_transforms)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    pin_memory=False, num_workers=num_workers)

    dl = cycle(dl)

    model = Classifier()
    loss_fn = module.CrossEntropyLoss()

    train_num_steps = 18750
    opt = SGD(parameters(model), lr=3e-3)

    step = 0
    pbar = tqdm(initial=step, total=train_num_steps, desc='Train')

    while step < train_num_steps:
        img, label = next(dl)
        img = Tensor(img)
        label = Tensor(label.numpy())

        probs = model(img)
        loss = loss_fn(probs, label)
        backward(loss)
        opt.update()
        opt.zero_grad()

        pbar.set_postfix(dict(
            loss=loss.data,
        ))

        step += 1
        pbar.update()
    pbar.close()

    # test part

    ds = torchvision.datasets.MNIST(
        'data', train=False, transform=image_transforms)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    pin_memory=False, num_workers=num_workers)
    total, crr = 0, 0
    for img, label in tqdm(dl, desc='Test'):
        img = Tensor(img)
        label = Tensor(label.numpy())
        pred = model.predict(img)
        total = total+label.data.shape[0]
        crr = crr+np.sum(label.data == pred.data)

    print(f'Acc on whole test set : {crr/total*100:.2f}%')
