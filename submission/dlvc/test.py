from abc import ABCMeta, abstractmethod

import numpy as np

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass

    @abstractmethod
    def __lt__(self, other) -> bool:
        '''
        Return true if this performance measure is worse than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass

    @abstractmethod
    def __gt__(self, other) -> bool:
        '''
        Return true if this performance measure is better than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass


class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self):
        '''
        Ctor.
        '''

        self.reset()

    def reset(self):
        '''
        Resets the internal state.
        '''
        self.acc = 0.0
        self.comparison = np.array([])


    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
            The predicted class label is the one with the highest probability.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        # check for right shape and values
        if (len(prediction.shape) != 2):
            raise ValueError('prediction must have 2 dimensions')
        if len(target.shape) != 1:
            raise ValueError('target must have 1 dimension')
        if target.shape[0] != prediction.shape[0]:
            raise ValueError('Target and prediction must have same nr of samples')
        
        if not (np.all((target >= 0) & (target <= prediction.shape[1]-1 ))):
            raise ValueError(f'Values must be between 0 and {prediction.shape[1]-1}')

        # select class label with highest probability per row
        predicted_class = prediction.argmax(axis = 1)

        # compare prediction with ground-truth
        # returns boolean array if label is equal or not
        self.comparison = np.equal(predicted_class, target)

        # update accuracy
        self.acc = self.accuracy()

    def __str__(self):
        '''
        Return a string representation of the performance.
        '''

        # return something like "accuracy: 0.395"
        return f'accuracy: {self.acc:.3f}'
    

    def __lt__(self, other) -> bool:
        '''
        Return true if this accuracy is worse than another one.
        Raises TypeError if the types of both measures differ.
        '''

        # See https://docs.python.org/3/library/operator.html for how these
        # operators are used to compare instances of the Accuracy class

        if type(self) != type(other):
            raise TypeError('types of both measures differ')
        
        if self.acc < other.acc:
            return True


    def __gt__(self, other) -> bool:
        '''
        Return true if this accuracy is better than another one.
        Raises TypeError if the types of both measures differ.
        '''

        if type(self) != type(other):
            raise TypeError('types of both measures differ')
        
        if self.acc > other.acc:
            return True
        

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        # TODO implement
        # on this basis implementing the other methods is easy (one line)
        if self.comparison.any():
            acc = np.sum(self.comparison) / self.comparison.shape[0]
        else:
            acc = 0.0

        return acc