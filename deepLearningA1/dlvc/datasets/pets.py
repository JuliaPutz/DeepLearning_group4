
from ..dataset import Sample, Subset, ClassificationDataset

class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    def __init__(self, fdir: str, subset: Subset):
        '''
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in BGR channel order.
        '''

        # TODO implement
        # See the CIFAR-10 website on how to load the data files
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        # filter out cat and dog images
        cat_label = 3
        dog_label = 5
        def filter_reshape(batch):
            data = batch[b'data']
            labels = batch[b'labels']

            labels_np = np.array(labels)
            idx = np.where((labels_np == cat_label) | (labels_np == dog_label))

            reshaped = data[idx].reshape(len(data[idx]),3,32,32)
            return(reshaped.transpose(0,2,3,1).astype("uint8"))

        try:
            #training
            if subset == Subset.TRAINING:
                #unpickle(fdir)
                print('select subset', 1)
            #validation
            elif subset == Subset.VALIDATION:
                batch_val = unpickle(fdir + 'data_batch_5')
                val_data = filter_reshape(batch_val)
            #test
            elif subset == Subset.TEST:
                batch_test = unpickle(fdir + 'test_batch')
                test_data = filter(batch_test)
        except ValueError:
            raise ValueError('fdir is not a Directory or file is missing')
        #pass

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        # TODO implement

        pass

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds. Negative indices are not supported.
        '''

        # TODO implement

        pass

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''

        # TODO implement

        pass
