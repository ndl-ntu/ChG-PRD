"""
Filename:        PRD_ReadoutLayer.py
Author:          Jia-Yi LI
Last Modified:   2024-12-20
Version:         1.0
Description:     Classification readout layer for MNIST/F-MNIST/HAR tasks.
                 Copyright (c) 2024 Nanyang Technological University
"""

import numpy as np
import misc
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

root_drc = "DIRC_TO_OUTPUT"

fashion_label = ['T-shirt/top', #0
                    'Trouser', #1
                    'Pullover', #2
                    'Dress', #3
                    'Coat', #4
                    'Sandal', #5
                    'Shirt', #6
                    'Sneaker', #7
                    'Bag', #8
                    'Ankle boot']#9
video_label = {'bend' : 0, #0
                    'jack' : 1, #1
                    'jump' : 2, #2
                    'run' : 3, #3
                    'side': 4, #4
                    'skip': 5, #5
                    'wave1': 6, #6
                    'wave2': 7,
                    'walk':8, #7
                    'pjump': 9}#8

def max_pool(img):
    """
    Applies max pooling to reduce the dimensions of the input images by a factor of 2.
    """
    print("Converting from 28x28 to 14x14")
    from skimage.measure import block_reduce

    block_size=(2,2)
    img_max_pool = np.empty((img.shape[0],int(img.shape[1]/block_size[0]),int(img.shape[2]/block_size[1])))
    for i in range(img.shape[0]):
        img_max_pool[i] = block_reduce(img[i], block_size, func=np.max)
    
    return img_max_pool

def avg_pool(img, block_size=(6,6)):
    """
    Applies average pooling to reduce the dimensions of the input images.
    """
    from skimage.measure import block_reduce
    img_avg_pool = np.empty((img.shape[0],int(img.shape[1]/block_size[0]),int(img.shape[2]/block_size[1])))
    for i in range(img.shape[0]):
        img_avg_pool[i] = block_reduce(img[i], block_size, func=np.average)
    
    return img_avg_pool

def readMnist(MNIST = 'fashion'):
    """
    Reads and loads the Fashion MNIST or MNIST dataset.
    """
    if MNIST == 'fashion': (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    elif MNIST == 'mnist': (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else: raise Exception("dataset not exist, choose from fashion or mnist")
    
    train_image = np.concatenate((X_train,X_test))
    train_label = np.concatenate((y_train,y_test))
    train_label.astype('int')

    return train_image, train_label

def datasetPrep(dataset):
    """
    Prepares the dataset by downsampling and normalizing the input images.
    """
    print("MNIST data downsampling.......")
    print(dataset.shape)
    image = dataset.reshape(70000, 28, 28)
    train_imgs_ds = max_pool(image)

    print("Converting from 8 bits to 1 bit depth")
    train_data = train_imgs_ds # train data shape (datasets_N,img_M*img_N)
    bit_depth = 255 # 16 for simple datasets, 255 for MNIST
    fac = 0.99 / bit_depth
    train_imgs = np.asfarray(train_data) * fac + 0.01

    train_imgs = train_imgs[:,:,1:13] 
    print(f"train_imgs shape {train_imgs.shape}")
    return train_imgs.reshape((70000, 14*12))

def loadExperiment(ex_id = 4096):
    """
    Loads experiment data from a specified file.
    """
    data_path = "Data/experiment/"
    ex_data = np.loadtxt(data_path + f"ex_{ex_id}.csv", 
                            delimiter=",")
    ex_data_normal  = ex_data.transpose()
    print(f"ex_data_normal shape {ex_data_normal.shape}")
    return ex_data_normal

def reservoirProcessIMAGE(train_imgs, ex_data_normal, node = 1, vector_out = False):
    """
    Processes images using reservoir computing.
    """
    from tqdm import tqdm
    img_M = int(14)
    img_N = int(12)
    datasets_N = train_imgs.shape[0]
    rc_mem = ex_data_normal.shape[0]
    rc_length = int(rc_mem/node)
    rc_length = 12
    print(rc_length)
    print("Reservoir computing (lineshape) using experimental reservoir computing devices")
    imgRC = np.zeros((datasets_N,int(img_M*img_N)), dtype=np.float32)
    for k in tqdm(range (datasets_N)):
        img = train_imgs[k]
        for i in range(0, img_M * img_N, rc_length):
            id = 0
            for id_idx in range(rc_length): 
                id += int(2**id_idx * img[i+rc_length-id_idx-1])
            for j in range (rc_length): 
                imgRC[k,i+j] = ex_data_normal[rc_mem-rc_length+j, id]
    imgRC = imgRC.reshape((imgRC.shape[0], img_M, img_N))
    imgRC_vec = np.zeros((datasets_N, img_M))
    for output in range(1,node+1):
        imgRC_vec += imgRC[:,:,int(output*11/(node))]
    print(f"Finished reservoir computing imgRC shape {imgRC_vec.shape}")
    if vector_out: imgRC_out = imgRC_vec
    else: imgRC_out = imgRC
    
    return imgRC_out

def train_set_gen(X, y, normalize = False):
    """
    Splits the data into training and testing sets.
    """
    if normalize: X = X/np.max(X)
    # generate train and test datasets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=1/7.0,
                                                    random_state=0
                                                    )
    return X_train, X_test, y_train, y_test

def ANN_mlp(imgRC = None, train_label = None, epochs=50):
    """
    Trains an ANN MLP model on reservoir processed data.
    """
    rc_shape = np.shape(imgRC.shape)[0]
    if rc_shape == 2: rc_shape=(28, ) # for vector input
    elif rc_shape == 3: rc_shape=(imgRC.shape[1], imgRC.shape[2]) # for video and image
    else: return -1
    
    X_train, X_test, y_train, y_test = train_set_gen(imgRC, train_label, normalize = False)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential([
        Flatten(input_shape=rc_shape),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_split=0.2, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    test_acc = np.round(test_acc,2)
    print(f'Test accuracy: {test_acc}')

    from sklearn.metrics import confusion_matrix
    # Make predictions
    y_pred = model.predict(X_test)

    # Convert predictions from one-hot encoded to class labels
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return(test_acc, history, model)

from Data.dataset.HAR import readWeizmannRevisited as HAR
def readHAR(frame_N = 8):
    """
    Reads and loads the HAR dataset.
    """
    (train_videos, train_labels) = HAR.ctDatasetTenClas(frames_per_clip = frame_N)
    return (train_videos, train_labels)

def reservoirProcess_HAR_1F(train_videos, ex_data_normal, node = 1):
    """
    Processes HAR dataset videos using reservoir computing.
    """
    height = 144
    width = 180
    rc_mem = int(np.log2(ex_data_normal.shape[1]))
    rc_length = int(rc_mem/node)
    num_test = ex_data_normal.shape[0]
    print(f"rc_length = {rc_length}, rc_mem = {rc_mem}")
    from tqdm import tqdm
    dataset_size = np.shape(train_videos)[0]
    imgRC = np.zeros((dataset_size, num_test, height * width), dtype=float)
    for i in tqdm(range (dataset_size)):
        frames = train_videos[i]
        img_id = 0

        for id_idx in range(rc_length): 
            img_id += 2**id_idx * frames[rc_length-id_idx-1]

        for idx in range (height * width):
            int_id= int(img_id[idx])
            for test in range (num_test):
                imgRC[i, test, idx] = ex_data_normal[test,int_id]
    return np.abs(imgRC)

def GausNoise(coef_var = 0, data = None, size = 1000):
    """
    Adds Gaussian noise to the input data.
    """
    data_min = np.min(data)
    data_max = np.max(data)
    data_noisy = np.zeros((data.shape[0], data.shape[1], size), dtype = np.float16)
    for i in range (data.shape[0]):
        for j in range(data.shape[1]):
            data_noisy[i,j] = np.random.normal(loc = data[i,j], scale = np.abs(coef_var * data[i,j]), size=size)
            data_noisy = np.clip(data_noisy, data_min/1000, data_max*1000)
    return data_noisy

def reservoirProcess_HAR_1F_noise(train_videos, ex_data_normal, node = 1):
    """
    Processes HAR dataset videos with Gaussian noise using reservoir computing.
    """
    height = 144
    width = 180
    rc_mem = int(np.log2(ex_data_normal.shape[1]))
    rc_length = int(rc_mem/node)
    num_test = ex_data_normal.shape[0]

    print(f"rc_length = {rc_length}, rc_mem = {rc_mem}")
    from tqdm import tqdm
    dataset_size = np.shape(train_videos)[0]
    imgRC = np.zeros((dataset_size, num_test, height * width), dtype=np.float16)

    gaus_size = ex_data_normal.shape[2]
    randi = np.random.randint(gaus_size, size = height * width)

    for i in tqdm(range (dataset_size)):
        frames = train_videos[i]
        img_id = 0

        for id_idx in range(rc_length): 
            img_id += 2**id_idx * frames[rc_length-id_idx-1]

        for idx in range (height * width):
            int_id= int(img_id[idx])
            for test in range (num_test):
                imgRC[i, test, idx] = ex_data_normal[test,int_id, randi[idx]]
    
    return np.abs(imgRC)

def rc_computing_video(ex_data, train_imgs, noise_rate):
    """
    Applies reservoir computing to video data with noise.
    """
    ex_data_normal = GausNoise(coef_var = noise_rate, data = ex_data, size = 1000)
    RC_output = reservoirProcess_HAR_1F_noise(train_imgs, ex_data_normal, node = 1)
    num_test = 5

    block_size = (6,6)
    video = RC_output
    video = np.reshape(video, (video.shape[0], video.shape[1], 144, 180))
    videos_rc = np.empty((video.shape[0], video.shape[1], int(144/block_size[0]), int(180/block_size[1])), dtype = np.float16)

    for i in range(num_test):
        videos_rc[:,i, :, :] = misc.pool(video[:,i, :, :], block_size, method = 'avg_pool', verbose=False)
    
    return videos_rc
