import numpy as np
import torch.nn as nn
import torch


from ..model import Model

class CnnClassifier(Model):
    '''
    Wrapper around a PyTorch CNN for classification.
    The network must expect inputs of shape NCHW with N being a variable batch size,
    C being the number of (image) channels, H being the (image) height, and W being the (image) width.
    The network must end with a linear layer with num_classes units (no softmax).
    The cross-entropy loss (torch.nn.CrossEntropyLoss) and SGD (torch.optim.SGD) are used for training.
    '''

    def __init__(self, net: nn.Module, input_shape: tuple, num_classes: int, lr: float, wd: float):
        '''
        Ctor.
        net is the cnn to wrap. see above comments for requirements.
        input_shape is the expected input shape, i.e. (0,C,H,W).
        num_classes is the number of classes (> 0).
        lr: learning rate to use for training (SGD with e.g. Nesterov momentum of 0.9).
        wd: weight decay to use for training.
        '''

        # Inside the train() and predict() functions you will need to know whether the network itself
        # runs on the CPU or on a GPU, and in the latter case transfer input/output tensors via cuda() and cpu().
        # To termine this, check the type of (one of the) parameters, which can be obtained via parameters() (there is an is_cuda flag).
        # You will want to initialize the optimizer and loss function here.
        # Note that PyTorch's cross-entropy loss includes normalization so no softmax is required

        self.net = net  # possibly check if net is valid (i.e. has correct input & output sizes)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        self.wd = wd
        self.is_cuda = self.net.parameters().is_cuda

        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, weigth_decay=self.wd)


    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple.
        '''
        return self.input_shape

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''
        return (self.num_classes,)

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns the training loss.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # TODO implement
        # Make sure to set the network to train() mode
        # See above comments on CPU/GPU

        if not isinstance(data, np.ndarray):
            raise TypeError(f"Input data is not an Array! Expecting np.ndarray, got {type(data)}.")
        if not isinstance(labels, np.ndarray):
            raise TypeError(f"Labels is not an Array! Expecting np.ndarray, got {type(labels)}.")
        
        if not data.dtype == np.float32:
            raise TypeError(f"Input data does not have the correct type! Expecting np.float32, got {data.dtype}.") 
        if not labels.dtype == int:
            raise TypeError(f"Labels do not have the correct type! Expecting int, got {labels.dtype}.")
        
        if data.shape[1:] != self.input_shape:
            raise ValueError(f"Input data shape is incorrect! Expecting {self.input_shape}, got {data.shape[1:]}.")
        if labels.shape[0] != data.shape[0]:
            raise ValueError(f"Labels and data are not the same length! Data: {data.shape[0]}, Labels: {labels.shape[0]}")
        if not ((max(labels) < self.num_classes) and (min(labels) >= 0)):
            raise ValueError(f"Labels have invalid values - found labels outside range of [0,{self.num_classes-1}].")  
        
        try:
            self.net.train()
            output = self.net(data)
            loss = self.loss(output, torch.from_numpy(labels))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss
        except:
            raise RuntimeError("Encountered an issue in training.")



    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        The scores are an array with shape (n, output_shape()).
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # Pass the network's predictions through a nn.Softmax layer to obtain softmax class scores
        # Make sure to set the network to eval() mode
        # See above comments on CPU/GPU

        if not isinstance(data, np.ndarray):
            raise TypeError(f"Input data is not an Array! Expecting np.ndarray, got {type(data)}.")
        if not data.dtype == np.float32:
            raise TypeError(f"Input data does not have the correct type! Expecting np.float32, got {data.dtype}.") 
        if data.shape[1:] != self.input_shape:
            raise ValueError(f"Input data shape is incorrect! Expecting {self.input_shape}, got {data.shape[1:]}.")
        
        try:
            softmax_layer = nn.Softmax(dim=1)
            self.net.eval()
            out = self.net(data)
            softmax_scores = softmax_layer(out)
            return softmax_scores
        except:
            raise RuntimeError("Encountered an issue in predicting.")
        

