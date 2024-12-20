"""
Filename:        MNISTandFMNIST_toPulse.py
Author:          Jia-Yi LI
Last Modified:   2024-12-20
Version:         1.0
Description:     Translate MNIST and F-MNIST to pulse input.
                 Copyright (c) 2024 Nanyang Technological University
"""

import numpy as np
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import matplotlib.pyplot as plt
import misc, os

saveScript = False
saveLabel = False
(X_train, y_train), (X_test, y_test) = mnist.load_data()
root_drc = os.path.abspath('DIRC_TO_PULSE_DATA_OUTPUT')

def max_pool(img):
    from skimage.measure import block_reduce

    block_size=(2,2)
    img_max_pool = np.empty((img.shape[0],int(img.shape[1]/block_size[0]),int(img.shape[2]/block_size[1])))
    for i in range(img.shape[0]):
        img_max_pool[i] = block_reduce(img[i], block_size, func=np.max)
    
    return img_max_pool

# Choose a class label
chosen_label = 2  # Change this to choose a different class label
indices = np.where(y_train == chosen_label)[0]
prng = np.random.RandomState(42)
random_index = prng.choice(indices)

# Original image
original_image = X_train[random_index]
# Max pooled image
downsampled_image = max_pool(X_train)[random_index]

train_X = max_pool(X_train)
test_X = max_pool(X_test)

train_X = np.reshape(train_X,(train_X.shape[0],train_X.shape[1]*train_X.shape[2]))
test_X = np.reshape(test_X,(test_X.shape[0],test_X.shape[1]*test_X.shape[2]))

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1]*X_test.shape[2]))

train_X, test_X = train_X / 255.0, test_X / 255.0
train_y, test_y = y_train, y_test

i = 0
id_mnist = 2405131220

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(np.concatenate((train_X, test_X)), 
                                                np.concatenate((train_y, test_y)).astype('int'),
                                                test_size=1/7.0,
                                                random_state=42)
X_train, X_test,_, _ = train_test_split(np.concatenate((X_train, X_test)), 
                                                np.concatenate((train_y, test_y)).astype('int'),
                                                test_size=1/7.0,
                                                random_state=42)
train_X, test_X, train_y, test_y = train_X[60*i:60*(i+1),:], test_X[10*i:10*(i+1),:],train_y[60*i:60*(i+1)], test_y[10*i:10*(i+1)]
X_train, X_test = X_train[60*i:60*(i+1),:], X_test[10*i:10*(i+1),:]


original = np.concatenate((X_train, X_test))
rc_input = np.concatenate((train_X,test_X))
rc_input = (rc_input-0.5)*6
print(original.shape, rc_input.shape)

data = np.concatenate((train_y,test_y))

if saveLabel == True:

    directory = "Script/Data/Label"
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, f"fmnist_70_label{i}_{id_mnist}.csv")
    np.savetxt(file_path, data, delimiter=",",fmt='%.3f')

rc_input = np.reshape(rc_input, (rc_input.shape[0], 14, 14))
prc_input = np.reshape(rc_input,(rc_input.shape[0]*rc_input.shape[1], rc_input.shape[2]))
data = prc_input
n = 0

if saveScript == True:

    directory = "Script/Data"
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, f"fmnist_70_{i}_{id_mnist}.csv")
    np.savetxt(file_path, data, delimiter=",",fmt='%.3f')
