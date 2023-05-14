import os
from collections import namedtuple

import cv2
import numpy as np
import torch

# A 2D vector. Used in Fn as an evaluation point.
Vec2 = namedtuple('Vec2', ['x1', 'x2'])

class AutogradFn(torch.autograd.Function):
    '''
    This class wraps a Fn instance to make it compatible with PyTorch optimizers
    '''
    @staticmethod
    def forward(ctx, fn, loc):
        ctx.fn = fn
        ctx.save_for_backward(loc)
        value = fn(Vec2(loc[0].item(), loc[1].item()))
        return torch.tensor(value)

    @staticmethod
    def backward(ctx, grad_output):
        fn = ctx.fn
        loc, = ctx.saved_tensors
        grad = fn.grad(Vec2(loc[0].item(), loc[1].item()))
        return None, torch.tensor([grad.x1, grad.x2]) * grad_output


def load_image(fpath: str) -> np.ndarray:
    '''
    Loads a 2D function from a PNG file and normalizes it to the interval [0, 1]
    Raises FileNotFoundError if the file does not exist.
    '''

    # not 100% sure is 2D Function a numpy array?
    if os.path.exists(fpath):
        image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        image_normalized = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return image_normalized
    else:
        raise FileNotFoundError(f"Path: {fpath} not found")
    

class Fn:
    '''
    A 2D function evaluated on a grid.
    '''

    def __init__(self, fn: np.ndarray, eps: float):
        '''
        Ctor that assigns function data fn and step size eps for numerical differentiation
        '''

        self.fn = fn
        self.eps = eps

    def visualize(self) -> np.ndarray:
        '''
        Return a visualization of the function as a color image. Use e.g. cv2.applyColorMap.
        Use the result to visualize the progress of gradient descent.
        '''

        # TODO implement
        vis = cv2.cvtColor(self.fn, cv2.COLOR_HSV2BGR)

        return vis

    def __call__(self, loc: Vec2) -> float:
        '''
        Evaluate the function at location loc.
        Raises ValueError if loc is out of bounds.
        '''

        # You can simply round and map to integers. If so, make sure not to set eps and learning_rate too low
        # Alternatively, you can implement some form of interpolation (for example bilinear)
        sx1_rounded = round(loc[0].item())
        sx2_rounded = round(loc[1].item())

        # raise Value error if loc is out of bounds 
        # values of loc need to be rounded first to be able to check that
        if sx1_rounded > self.fn.shape[0] or sx2_rounded > self.fn.shape[1]:
            raise ValueError("loc is out of bounds")

        return self.fn[sx1_rounded, sx2_rounded]
        

    def grad(self, loc: Vec2) -> Vec2:
        '''
        Compute the numerical gradient of the function at location loc, using the given epsilon.
        Raises ValueError if loc is out of bounds of fn or if eps <= 0.
        '''

        # TODO implement

        pass

if __name__ == '__main__':
    # Parse args
    import argparse

    parser = argparse.ArgumentParser(description='Perform gradient descent on a 2D function.')
    parser.add_argument('fpath', help='Path to a PNG file encoding the function')
    parser.add_argument('sx1', type=float, help='Initial value of the first argument')
    parser.add_argument('sx2', type=float, help='Initial value of the second argument')
    parser.add_argument('--eps', type=float, default=1.0, help='Epsilon for computing numeric gradients')
    parser.add_argument('--learning_rate', type=float, default=10.0, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0, help='Beta parameter of momentum (0 = no momentum)')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
    args = parser.parse_args()

    # Init
    image_fn = load_image(args.fpath)
    fn = Fn(image_fn, args.eps)
    vis = fn.visualize()

    # PyTorch uses tensors which are very similar to numpy arrays but hold additional values such as gradients
    loc = torch.tensor([args.sx1, args.sx2], requires_grad=True)
    optimizer = torch.optim.SGD([loc], lr=args.learning_rate, momentum=args.beta, nesterov=args.nesterov)

    # Find a minimum in fn using a PyTorch optimizer
    # See https://pytorch.org/docs/stable/optim.html for how to use optimizers
    while True:
        # Visualize each iteration by drawing on vis using e.g. cv2.line()
        # Find a suitable termination condition and break out of loop once done

        # This returns the value of the function fn at location loc.
        # Since we are trying to find a minimum of the function this acts as a loss value.
        # loss = AutogradFn.apply(fn, loc)

        cv2.imshow('Progress', vis)
        cv2.waitKey(50)  # 20 fps, tune according to your liking
