from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.batches import BatchGenerator
from dlvc.test import Accuracy
import dlvc.ops as ops
import numpy as np
import torch


# TODO: Define the network architecture of your linear classifier.
class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim, num_classes):
    super(LinearClassifier, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, num_classes)

  def forward(self, x):
    tensor = torch.from_numpy(x)
    out = self.linear1(tensor)
    return out

# load the data
print('Load Data ...')
fdir = r"cifar\\"
train = PetsDataset(fdir = fdir, subset = Subset.TRAINING)
test = PetsDataset(fdir = fdir, subset = Subset.TEST)
validate = PetsDataset(fdir = fdir, subset = Subset.VALIDATION)

print('generate batches ....')
# TODO: Create a 'BatchGenerator' for training, validation and test datasets.
op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])
batchtrain = BatchGenerator(train, num = len(train), shuffle=False, op = op)
batchtest = BatchGenerator(test, num = len(test), shuffle=False, op = op)
batchval = BatchGenerator(validate, num = len(validate), shuffle=False, op = op)


# TODO: Create the LinearClassifier, loss function and optimizer. 
model = LinearClassifier(32*32*3, train.num_classes())
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


'''
TODO: Train a model for multiple epochs, measure the classification accuracy on the validation dataset throughout the training and save the best performing model. 
After training, measure the classification accuracy of the best perfroming model on the test dataset. Document your findings in the report.
'''
print('train classifier ...')
def train_test_classifier(epochs: int):
  b_train, = batchtrain
  b_test, = batchtest
  b_val, = batchval
  accuracy = Accuracy()
  best_acc = Accuracy()
  best_model = None
  acc_log = np.array(['epoch', 'val accuracy'])

  for epoch in range(epochs):
      
      # train model
      output = model(b_train.data)
      loss = criterion(output, torch.from_numpy(b_train.label))
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      # validate model
      out_val = model(b_val.data)
      accuracy.update(out_val.detach().numpy(), b_val.label)
      acc_log = np.vstack([acc_log, [str(epoch+1), str(accuracy.acc)]])
      if accuracy > best_acc:
        best_acc.update(out_val.detach().numpy(), b_val.label)
        best_model = model.state_dict()

      print(f'epoch {epoch+1}')
      print(f'train loss: {loss.item():.3f}')
      print(f'val {accuracy}')
      print('-------------------------------------')

  # load and test best model
  model.load_state_dict(best_model)
  out_test = model(b_test.data)
  accuracy.update(out_test.detach().numpy(), b_test.label)
  acc_log = np.vstack([acc_log, ['final test accuracy', str(accuracy.acc)]])


  print('-------------------------------------')
  print(f'best val {best_acc}')
  print(f'test {accuracy}')
  np.savetxt("accuracies.csv", acc_log, delimiter = ",", fmt='%s')
  
# run for 100 epochs
train_test_classifier(100)