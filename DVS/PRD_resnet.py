"""
Filename:        PRD_resnet.py
Author:          Xiao-Yu LIN
Last Modified:   2024-12-20
Version:         1.0
Description:     Classification readout layer for DVS tasks.
                 Copyright (c) 2024 Nanyang Technological University
"""

import numpy as np
import IPython
from keras import Input

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Layer, Add, Dropout
from keras.models import Model
import tensorflow as tf
from keras.regularizers import l2
from DVS.PRD_event_stream import n_num


class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, k: float, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]
        self.__k = k

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)
            res = self.__k * res

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class None_ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__strides = [2, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        out = tf.nn.relu(x)
        return out


class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)

        # control current simultaneously
        filters = [32, 64, 64]
        k = [0.8, 0.9, 1, 1, 1, 1]
        # k = [1, 1, 1, 1, 1, 1]
        self.conv_1 = Conv2D(filters[0],
                             (4, 4),
                             strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(
            pool_size=(2, 2),
            strides=2,
            padding="same")

        self.res_1_1 = ResnetBlock(filters[0], k[0])
        self.res_1_2 = ResnetBlock(filters[0], k[1])

        self.res_2_1 = ResnetBlock(filters[1], k[1], down_sample=True)
        self.res_2_2 = ResnetBlock(filters[1], k[2])

        self.res_3_1 = ResnetBlock(filters[2], k[4], down_sample=True)
        self.res_3_2 = ResnetBlock(filters[2], k[5])

        self.res_4_1 = ResnetBlock(filters[2], k[4], down_sample=True)
        self.res_4_2 = ResnetBlock(filters[2], k[5])

        self.avg_pool = GlobalAveragePooling2D()
        self.merge = Add()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")
        # self.fc = Dense(10, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)

        # with extra shortcut
        out_0 = self.pool_2(out)
        out = self.res_1_1(out_0)
        out = self.res_1_2(out)
        out = self.merge([out, out_0])
        out = self.res_2_1(out)

        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

    def build_graph(self):
        x = Input(shape=(128, 128, 2))
        return Model(inputs=[x], outputs=self.call(x))


# callback to clear the training output
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


# #########build model############################################
def hyper_tuner(aug, tune):
    import time
    start = time.time()

    if tune:
        from improvement_tuning import load_tuned_removed7
        x_train, x_test, y_train, y_test, x_val, y_val = load_tuned_removed7(Aug=aug)
    else:
        from DVS.frames_processing import polar_remove_load, polar_remove_set7_load
        x_train, x_test, y_train, y_test, x_val, y_val = polar_remove_set7_load(aug)


    print('train shape{0}, validation shape {1},test shape {2}'.format
          (x_train.shape, x_val.shape, x_test.shape))
    # ########residual##########################
    from keras.callbacks import EarlyStopping
    hypermodel = ResNet18(10)
    # #print the model# ############
    hypermodel.build(input_shape=(None, 128, 128, 2))
    hypermodel.build_graph().summary()
    tf.keras.utils.plot_model(
        hypermodel.build_graph(),  # here is the trick (for now)
        to_file='model.png', dpi=96,  # saving
        show_shapes=True, show_layer_names=True,  # show shapes and layer name
        expand_nested=False  # will show nested block
    )
    # ################## learning rate scheduler
    initial_learning_rate = 0.01

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,

        decay_steps=5000,
        decay_rate=0.5,

        staircase=True)
    # #########################
    hypermodel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        # optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"])
    # hypermodel.summary()
    # ############################# call back: the early stopping
    es = EarlyStopping(patience=20, restore_best_weights=True, monitor="val_accuracy")
    # pationce = 20
    from keras.callbacks import ReduceLROnPlateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10,
        min_lr=1e-6,
        verbose=1)
    # ####check point path for model loading

    from DVS.PRD_event_stream import find_path
    root_dir, train_path, test_path = find_path()
    #     checkpoint_path = root_dir + "/dvs_SAVE_new_2/cp.ckpt"
    checkpoint_path = root_dir + "/dvs_SAVE_oct9_2/cp.ckpt"
    import os
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    STEPS = 96
    bs = int(len(x_train) / STEPS)
    history = hypermodel.fit(x_train, y_train, batch_size=bs, steps_per_epoch=STEPS,
                             # epochs=140,
                             epochs=120,
                             validation_data=(x_val, y_val),
                             callbacks=[
                                 es,
                                 cp_callback,
                             ]
                             )
    # # Create a callback that saves the model's weights###########
    hypermodel.summary()
    temp = np.array([history.history['loss'],history.history['val_loss'],history.history['accuracy'],history.history['val_accuracy']])
    np.savetxt('C:/Users/ASUS/Desktop/history.csv', temp.swapaxes(0,1), delimiter=",", fmt="%.3f")
    accuracy = hypermodel.evaluate(x_test, y_test, verbose=0)[1]
    print('Accuracy:', accuracy)
    y_pred = np.array(list(map(lambda x: np.argmax(x), hypermodel.predict(x_test))))

    end = time.time()
    print('Training time: ', end - start)
    return x_train, x_val, x_test, y_train, y_val, y_test


# supplementary###########################################################
# takes the most frequent class in 5 frames(which belongs to the same event stream)
def sum_n_acc(y_pred, test_label_list, n_num):
    import statistics
    y_pred_1 = []
    y_test_1 = []
    for i in range(24 * 10):
        y_pred_t = y_pred[0 + n_num * i: n_num - 1 + n_num * i]
        y_pred_1.append(statistics.mode(y_pred_t))
        y_test_t = test_label_list[0 + n_num * i: n_num * (i + 1) - 1]
        y_test_1.append(statistics.mode(y_test_t))
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test_1, y_pred_1)
    classification_report(y_test_1, y_pred_1)
    accuracy = sum(1 for x, y in zip(y_test_1, y_pred_1) if x == y) / len(y_test_1)
    print('final accuracy is: ', accuracy)


# #############call back. load model from check point############
def load_cp():
    import time
    start = time.time()
    from DVS.frames_processing import polar_remove_load, polar_remove_set7_load
    train_n, test_n, train_label_list, test_label_list = polar_remove_set7_load(aug=True)
    from sklearn.utils import shuffle
    train_n, train_label_list = shuffle(train_n, train_label_list, random_state=15)
    from sklearn.model_selection import train_test_split
    train_n, x_val, train_label_list, y_val = (
        train_test_split(train_n, train_label_list, test_size=0.125, random_state=1))
    # #######################################residual##########################
    from keras.callbacks import EarlyStopping
    hypermodel = ResNet18(10)

    # #print the model# ############
    hypermodel.build(input_shape=(None, 128, 128, 2))
    hypermodel.build_graph().summary()
    tf.keras.utils.plot_model(
        hypermodel.build_graph(),  # here is the trick (for now)
        to_file='model.png', dpi=96,  # saving
        show_shapes=True, show_layer_names=True,  # show shapes and layer name
        expand_nested=False  # will show nested block
    )
    hypermodel.compile(
        optimizer=tf.keras.optimizers.Adam(),
        # optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"])

    from DVS.PRD_event_stream import find_path
    root_dir, train_path, test_path = find_path()
    checkpoint_path = root_dir + "training_test_8/cp.ckpt"
    # Loads the weights
    hypermodel.load_weights(checkpoint_path)

    # Re-evaluate the model
    loss, acc = hypermodel.evaluate(test_n, test_label_list, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    accuracy = hypermodel.evaluate(test_n, test_label_list, verbose=0)[1]
    print('Accuracy:', accuracy)
    y_pred = np.array(list(map(lambda x: np.argmax(x), hypermodel.predict(test_n))))

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(test_label_list, y_pred)
    classification_report(test_label_list, y_pred)

    # #########
    sum_n_acc(y_pred, test_label_list, n_num)
    end = time.time()
    print('Training time: ', end - start)
    return train_n, x_val, test_n, train_label_list, y_val, test_label_list
