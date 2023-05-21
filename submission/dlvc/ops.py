from typing import List, Callable

import numpy as np

# All operations are functions that take and return numpy arrays
# See https://docs.python.org/3/library/typing.html#typing.Callable for what this line means
Op = Callable[[np.ndarray], np.ndarray]

def chain(ops: List[Op]) -> Op:
    '''
    Chain a list of operations together.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        for op_ in ops:
            sample = op_(sample)
        return sample

    return op

def type_cast(dtype: np.dtype) -> Op:
    '''
    Cast numpy arrays to the given type.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample.astype(dtype)
    
    return op

def vectorize() -> Op:
    '''
    Vectorize numpy arrays via "numpy.ravel()".
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.ravel(sample)
    
    return op

def add(val: float) -> Op:
    '''
    Add a scalar value to all array elements.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample + val
    
    return op

def mul(val: float) -> Op:
    '''
    Multiply all array elements by the given scalar.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample * val
    
    return op

def hwc2chw() -> Op:
    '''
    Flip a 3D array with shape HWC to shape CHW.
    '''
    def op(sample: np.ndarray) -> np.ndarray:
        try:
            return np.transpose(sample, (2,0,1))
        except ValueError:
            print(sample.shape)
            raise
    return op

def hflip() -> Op:
    '''
    Flip arrays with shape HWC horizontally with a probability of 0.5.
    '''
    def op(sample: np.ndarray) -> np.ndarray:
        if np.random.randint(2): # 50% chance to be 0 and thus False
            return np.flip(sample, 1)
        else:
            return sample
    return op

def rcrop(sz: int, pad: int, pad_mode: str) -> Op:
    '''
    Extract a square random crop of size sz from arrays with shape HWC.
    If pad is > 0, the array is first padded by pad pixels along the top, left, bottom, and right.
    How padding is done is governed by pad_mode, which should work exactly as the 'mode' argument of numpy.pad.
    Raises ValueError if sz exceeds the array width/height after padding.
    '''
    def op(sample: np.ndarray) -> np.ndarray:
        padding = ((pad,pad),(pad,pad),(0,0)) # pad top/bottom and left/right for first two axes, but not the last (=channels)
        padded = np.pad(sample, padding, pad_mode)
        if sz > min(padded.shape[:-1]):
            raise ValueError("Crop size is larger than padded image size!")
        tlc = np.random.randint((padded.shape[0]-sz, padded.shape[1]-sz)) # pick top left corner of crop so that at least sz pixels remain
        return padded[tlc[0]:tlc[0]+sz,tlc[1]:tlc[1]+sz] # crop by slicing from top left corner
    return op

# extra augmentation method
def rotate(num = None) -> Op:
    '''
    Rotate the sample by 90 degrees "num" times. If num is None, a random value is picked for each image.
    The sample array has shape HWC and will be rotated in the plane of the first two axes. 
    '''
    def op(sample: np.ndarray) -> np.ndarray:
        if num is None:
            num = np.random.randint(4) # equal chance for 0, 90, 180 and 270 degree rotations
        elif not isinstance(num, int):
            raise ValueError("num needs to be an integer or None.")
        return np.rot90(sample, num, axes=(0,1))
    return op