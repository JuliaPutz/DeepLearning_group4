from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.batches import BatchGenerator
from dlvc.test import Accuracy
from dlvc.models.pytorch import CnnClassifier
import dlvc.ops as ops

import numpy as np
import torch.nn as nn
import torch


class CNNforCatsAndDogs(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNforCatsAndDogs, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # halves size to 16x16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # halves size again to 8x8
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(32*(input_dim //4)*(input_dim//4), 128), # after 2 rounds of MaxPooling, the output size is 8x8 with 32 channels
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # flatten tensor
        x = self.linear_layers(x)
        return x


# load the data
print('Loading Data...')
fdir = r"../cifar"
train = PetsDataset(fdir = fdir, subset = Subset.TRAINING)
test = PetsDataset(fdir = fdir, subset = Subset.TEST)
validate = PetsDataset(fdir = fdir, subset = Subset.VALIDATION)

# specificy batch operations
op = ops.chain([
    ops.hwc2chw(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])

# make batches
batchtrain = BatchGenerator(train, num = 128, shuffle=False, op = op)
batchtest = BatchGenerator(test, num = 128, shuffle=False, op = op)
batchval = BatchGenerator(validate, num = 128, shuffle=False, op = op)

# define CNN 
model = CNNforCatsAndDogs(32, train.num_classes())
if torch.cuda.is_available():
    print("Using cuda!")
    model = model.cuda()
else:
    print("Using CPU.")
    model = model.cpu()
clf = CnnClassifier(model, (3,32,32), 2, 0.02, 0.005)


accuracy = Accuracy()
best_acc = Accuracy()
best_model = None
acc_log = np.array(['epoch', 'val_accuracy'])
min_train_loss = 1

for epoch in range(100):
    train_losses = []
    for b_train in batchtrain:
        train_loss = clf.train(b_train.data, b_train.label)
        train_losses.append(train_loss.numpy())
    train_losses = np.array(train_losses)
    if np.mean(train_losses) < min_train_loss:
        min_train_loss = np.mean(train_losses)

    for b_val in batchval:
        out_val = clf.predict(b_val.data)
        accuracy.update(out_val.numpy(), b_val.label)
    acc_log = np.vstack([acc_log, [str(epoch+1), str(accuracy.acc)]])

    if accuracy > best_acc:
        best_acc.update(out_val.numpy(), b_val.label)
        best_model = clf.net.state_dict()

    print(f'epoch {epoch+1}')
    print(f'train loss: {np.mean(train_losses):.3f} ± {np.std(train_losses):.3f}')
    print(f'val {accuracy}')
    print('-------------------------------------')

# load best model and test it
model.load_state_dict(best_model)
final_clf = CnnClassifier(model, (3,32,32), 2, 1e-4, 0.2)
for b_test in batchtest:
    out_test = final_clf.predict(b_test.data)
    accuracy.update(out_test.numpy(), b_test.label)
acc_log = np.vstack([acc_log, ['final test accuracy', str(accuracy.acc)]])

print('-------------------------------------')
print(f'min train loss: {min_train_loss:.3f}')
print(f'best val {best_acc}')
print(f'test {accuracy}')
np.savetxt("accuracies_cnn.csv", acc_log, delimiter = ",", fmt='%s')
