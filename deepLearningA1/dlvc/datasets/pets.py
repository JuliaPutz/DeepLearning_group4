
from ..dataset import Sample, Subset, ClassificationDataset
import numpy as np

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
        #######################################
        # check channel order
        # load train data and stack together
        # len function
        # getitem function
        # num classes
        #######################################

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

            reshaped = data[idx].reshape(len(data[idx]),3,32,32).transpose(0,2,3,1).astype("uint8") # image are brought into the (32,32,3) shape
            bgr = reshaped[..., ::-1] # convert images from initial rgb to bgr
            renamed = (labels_np[idx]==dog_label).astype(int)   # dog is 1, cat is 0
            return bgr, renamed


        # referring to subsets via their number is allowed, as a treat
        if not isinstance(subset, Subset):
            self.subset = Subset(subset)
        else:
            self.subset = subset
        self.num_classes = 2
        self.data = None
        self.labels = None

        if fdir[-1] != "/":
            fdir = fdir + "/"


        try:
            #training
            if self.subset == Subset.TRAINING:
                data_holder, labels_holder = [],[]
                for i in [1,2,3,4]:
                    batch = unpickle(fdir + f'data_batch_{i}')
                    d,l = filter_reshape(batch)
                    data_holder.append(d)
                    labels_holder.append(l)
                self.data = np.concatenate(data_holder, 0, dtype="uint8")
                self.labels = np.concatenate(labels_holder, 0)

            #validation
            elif self.subset == Subset.VALIDATION:
                batch_val = unpickle(fdir + 'data_batch_5')
                self.data, self.labels = filter_reshape(batch_val)

            #test
            elif self.subset == Subset.TEST:
                batch_test = unpickle(fdir + 'test_batch')
                self.data, self.labels = filter_reshape(batch_test)
            else:
                raise ValueError("Not a valid ")
        except ValueError:
            raise ValueError('fdir is not a Directory or file is missing')
        
        self.length = self.data.shape[0]
        

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''
        return self.length


    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds. Negative indices are not supported.
        '''
        if idx < 0 or idx >= self.length:
            raise IndexError
        return Sample(idx, self.data[idx], self.labels[idx])

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''
        return len(np.unique(self.labels))