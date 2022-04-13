import random
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
import time
import math
import numpy as np
import torch
from torch.utils import data


###################################################################################
### VISUALIZATION
###################################################################################


def use_svg_display():
    """
    Use the SVG format to display plot in jupyter
    """
    backend_inline.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    """
    Set the figure size for matplotlib
    """
    use_svg_display()
    plt.rcParams["figure.figsize"] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def has_one_axis(x):
    return (
        hasattr(x, "ndim")
        and x.ndim == 1
        or isinstance(x, list)
        and not hasattr(x[0], "__len__")
    )


def plot(
    xb,
    yb=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    xlim=None,
    ylim=None,
    xscale="linear",
    yscale="linear",
    fmts=("-", "m--", "g-.", "r:"),
    figsize=(3.5, 2.5),
    axes=None,
):
    """Plot data points"""
    ## step-1: check for the LEGEND
    if legend is None:
        legend = []

    ## step-2: set the FIGSIZE
    set_figsize(figsize)

    ## step-3: check axes
    axes = axes if axes else plt.gca()

    ## step-4: check "x"
    if has_one_axis(xb):
        xb = [xb]

    ## step-5: check "y"
    if yb is None:
        xb, yb = [[]] * len(xb), xb
    elif has_one_axis(yb):
        yb = [yb]

    ## step-6: align x & y
    if len(xb) != len(yb):
        xb = xb * len(yb)

    ## step-7:
    axes.cla()

    ## step-8
    for x, y, fmt in zip(xb, yb, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)

    ## step-9:
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def show_images(images, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if torch.is_tensor(img):
            ## a tensor image....so convert it to numpy image
            ax.imshow(img.numpy())
        else:
            ## a PIL image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


###################################################################################
### BENCHMARKING
###################################################################################


class Timer:
    """Record multiple running times"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer"""
        self.tik = time.time()

    def stop(self):
        """Stop the timer"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average times"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of all times"""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time"""
        return np.array(self.times).cumsum().tolist()


###################################################################################
### DATA
###################################################################################


def synthetic_data(w, b, num_examples):
    """Genrate y = Wx + b + noise"""
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b  ## y = x@w + b
    y += torch.normal(0, 0.01, y.shape)  ## y = x@w + b + noise
    return x, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    """Iterate through the whole dataset consisting of (features, labels) with batch-wise random smapling"""
    num_examples = len(features)
    idxs = list(range(num_examples))
    ## randomly sample from the indices
    random.shuffle(idxs)
    for i in range(0, num_examples, batch_size):
        batch_idxs = torch.tensor(idxs[i : min(i + batch_size, num_examples)])
        yield features[batch_idxs], labels[batch_idxs]


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


### FashionMNIST
def get_fashion_mnist_labels(labels):
    """Returns text labels for the FasshionMNIST dataset"""
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [text_labels[i] for i in labels]


###################################################################################
### MODELS
###################################################################################


def linear_regression(x, w, b):
    return torch.matmul(x, w) + b


###################################################################################
### LOSS
###################################################################################


def squared_loss(y_hat, y):
    """Squared Loss"""
    y = y.reshape(y_hat.shape)
    return (1 / 2) * (y_hat - y) ** 2


###################################################################################
### OPTIMIZERS
###################################################################################


def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent"""
    with torch.no_grad():
        for p in params:
            ## here we need to divide by "batch size"
            ## because during forward prop, the gradient has accumulated for the mini-batch
            p -= lr * p.grad / batch_size
            p.grad.zero_()  ## zero the gradient
