import matplotlib.pyplot as plt
import numpy as np

fdir = r"cifar\\"

# function to unpack datasets
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#### TEST ####
batch = unpickle(fdir + 'test_batch')

print(batch.keys())
data = batch[b'data']

print(len(data))

reshaped = data.reshape(len(data),3,32,32)
tr = reshaped.transpose(0,2,3,1).astype("uint8")

plt.imshow(tr[9999].astype("uint8"))
plt.show()

print(tr.shape)
print(type(tr))

#### VALIDATION ####
print('--------validation-------------')
fdir = r"E:\\TU\\deepLearning\\A1\\cifar\\"
batch_val = unpickle(fdir + 'data_batch_5')

print(batch.keys())
data_val = batch_val[b'data']

print(len(data_val))

reshaped_val = data_val.reshape(len(data_val),3,32,32)
tr_val = reshaped_val.transpose(0,2,3,1).astype("uint8")

plt.imshow(tr_val[2].astype("uint8"))
plt.show()

print(tr_val.shape)
print(type(tr_val))