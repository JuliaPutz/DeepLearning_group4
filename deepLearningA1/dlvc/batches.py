import typing
import numpy as np

from .dataset import Dataset
from .ops import Op

class Batch:
    '''
    A (mini)batch generated by the batch generator.
    '''

    def __init__(self, data, label, idx):
        '''
        Ctor.
        '''

        self.data = data
        self.label = label
        self.idx = idx


class BatchGenerator:
    '''
    Batch generator.
    Returned batches have the following properties:
      data: numpy array holding batch data of shape (s, SHAPE_OF_DATASET_SAMPLES).
      label: numpy array holding batch labels of shape (s, SHAPE_OF_DATASET_LABELS).
      idx: numpy array with shape (s,) encoding the indices of each sample in the original dataset.
    '''

    def __init__(self, dataset: Dataset, num: int, shuffle: bool, op: Op=None):
        '''
        Ctor.
        Dataset is the dataset to iterate over.
        num is the number of samples per batch. the number in the last batch might be smaller than that.
        shuffle controls whether the sample order should be preserved or not.
        op is an operation to apply to input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values, such as if num is > len(dataset).
        '''

        # TODO:
        # Which type has op???

        # simple non-operation for when no op argument is passed
        def noOp(sample: np.ndarray) -> np.ndarray:
            return sample

        # check the argument types and raise TypeError if invalid
        if not (isinstance(num, int) and 
                isinstance(dataset, Dataset) and
                isinstance(shuffle, bool)):
            raise TypeError('Please check your argument types')
        # check argument values and raise ValueError if invalid
        elif (num > len(dataset)):
            raise ValueError('Num cannot be bigger then the dataset')
        
        self.dataset = dataset
        self.batch_size = num
        self.shuffle = shuffle
        self.op = op if op is not None else noOp
        self.batch_data_shape = self.op(self.dataset[0].data).shape

        # generate an array of valid indices, assuming dataset indices always 
        self.valid_indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.valid_indices)

    def __len__(self) -> int:
        '''
        Returns the total number of batches the dataset is split into.
            This is identical to the total number of batches yielded every time the __iter__ method is called.
        '''

        # if the batch size perfectly divides the data set length there is no vestigal last batch, hence plusone adds 0 if the modulo is 0
        plusone = len(self.dataset) % self.batch_size != 0
        return len(self.dataset) // self.batch_size + int(plusone)

    def __iter__(self) -> typing.Iterable[Batch]:
        '''
        Iterate over the wrapped dataset, returning the data as batches.
        '''

        def get_batch(i_start: int, i_end: int) -> Batch:
            batch_indices = self.valid_indices[i_start:i_end]
            bsize = len(batch_indices)
            data = np.zeros((bsize, *self.batch_data_shape))
            labels = np.zeros(bsize)
            for j in range(i_start, i_end):
                data_item = self.dataset[batch_indices[j]]
                data[j] = self.op(data_item.data)
                labels[j] = data_item.label
            
            return Batch(idx = batch_indices, data = data, label = labels)

        i = 0
        while i < len(self.dataset) - self.batch_size:
            yield get_batch(i, i+self.batch_size)
            i += self.batch_size

        if i < len(self.dataset):
            yield get_batch(i, len(self.dataset))
