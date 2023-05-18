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
        # output dtype needs to be float not int to properly represent image
        # in range [0, 1]
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
        # ?? How does this work with normalized input with dtype CV_32F ??
        # this might be a workaround, looks right at least
        vis = cv2.applyColorMap(np.uint8(self.fn * 255), cv2.COLORMAP_BONE)

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

        return self.fn[sx2_rounded, sx1_rounded]
        

    def grad(self, loc: Vec2) -> Vec2:
        '''
        Compute the numerical gradient of the function at location loc, using the given epsilon.
        Raises ValueError if loc is out of bounds of fn or if eps <= 0.
        '''

        # implemented
        # are errors correct?
        if loc[0] > self.fn.shape[0] or loc[1] > self.fn.shape[1]:
            raise ValueError("loc is out of bounds")
        elif self.eps <= 0:
            raise ValueError("eps has to be > 0")
        elif loc[0] + self.eps > self.fn.shape[0] or loc[1] + self.eps > self.fn.shape[1]:
            raise ValueError("eps + loc is out of bounds")
        else:
            position_loc = self(loc)
            position_plus_x1 = self(Vec2(loc[0] + self.eps, loc[1]))
            position_plus_x2 = self(Vec2(loc[0], loc[1] + self.eps))

            num_gradient_x1 = (position_plus_x1 - position_loc) / self.eps
            num_gradient_x2 = (position_plus_x2 - position_loc) / self.eps

        return Vec2(num_gradient_x1, num_gradient_x2)


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
        loss = AutogradFn.apply(fn, loc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate starting and end point of line representing the gradient
        start = (round(loc[0].item()), round(loc[1].item()))
        end = (round(loc.grad[0].item() * 2000) + round(loc[0].item()),  round(loc.grad[1].item() * 2000) + round(loc[1].item()))
    
        cv2.line(vis, start, end, [0, 255, 0], 2)

        cv2.imshow('Progress', vis)
        cv2.waitKey(50)  # 20 fps, tune according to your liking

        if loc.grad[0].item() == 0. and loc.grad[1].item() == 0.:
            break