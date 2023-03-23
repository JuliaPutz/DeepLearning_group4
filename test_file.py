import matplotlib.pyplot as plt
import numpy as np

fdir = r"cifar\\"
cat_label = 3
dog_label = 5

#check label names/classes
#batch_meta = unpickle(fdir + 'batches.meta')
#print(batch_meta)

# function to unpack datasets
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# filter out cat and dog images
def filter_reshape(batch):
    data = batch[b'data']
    labels = batch[b'labels']

    labels_np = np.array(labels)
    idx = np.where((labels_np == cat_label) | (labels_np == dog_label))

    reshaped = data[idx].reshape(len(data[idx]),3,32,32)
    return(reshaped.transpose(0,2,3,1).astype("uint8"))

#### TEST ####
print('----------TEST-------------')
batch = unpickle(fdir + 'test_batch')

test_data = filter_reshape(batch)

print(test_data.shape)
print(type(test_data))
print(len(test_data))
plt.imshow(test_data[0].astype("uint8"))
plt.show()


#### VALIDATION ####
print('--------validation-------------')
batch_val = unpickle(fdir + 'data_batch_5')

val_data = filter_reshape(batch_val)

print(val_data.shape)
print(type(val_data))
print(len(val_data))
plt.imshow(val_data[2].astype("uint8"))
plt.show()
