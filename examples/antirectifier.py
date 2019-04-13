'''
#This example demonstrates how to write custom layers for Keras.

We build a custom activation layer called 'Antirectifier',
which modifies the shape of the tensor that passes through it.
We need to specify two methods: `compute_output_shape` and `call`.

Note that the same result can also be achieved via a Lambda layer.

Because our custom layer is written with primitives from the Keras
backend (`K`), our code can run both on TensorFlow and Theano.

60000/60000 [==============================] - 2s 36us/step - loss: 0.0050 - acc: 0.9984 - val_loss: 0.0927 - val_acc: 0.9821
Time taken: 0:01:33.612851 by RTX2070

60000/60000 [==============================] - 9s 148us/step - loss: 0.6021 - acc: 0.9137 - val_loss: 0.1515 - val_acc: 0.9613
Epoch 2/40 other video card
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras import layers
from keras.datasets import mnist
from keras import backend as K

from datetime import datetime
startTime = datetime.now()

# http://www.itdaan.com/keywords/Tensorflow+%E6%8A%A5%E9%94%99+CUBLAS+STATUS+ALLOC+FAILED.html
# import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
# http://www.itdaan.com/blog/2018/03/27/39f85117d9f44abb157bdc199174177d.html
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.233)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


class Antirectifier(layers.Layer):
    '''This is the combination of a sample-wise
    L2 normalization with the concatenation of the
    positive part of the input with the negative part
    of the input. The result is a tensor of samples that are
    twice as large as the input samples.

    It can be used in place of a ReLU.

    # Input shape
        2D tensor of shape (samples, n)

    # Output shape
        2D tensor of shape (samples, 2*n)

    # Theoretical justification
        When applying ReLU, assuming that the distribution
        of the previous output is approximately centered around 0.,
        you are discarding half of your input. This is inefficient.

        Antirectifier allows to return all-positive outputs like ReLU,
        without discarding any data.

        Tests on MNIST show that Antirectifier allows to train networks
        with twice less parameters yet with comparable
        classification accuracy as an equivalent ReLU-based network.
    '''

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] *= 2
        return tuple(shape)

    def call(self, inputs):
        inputs -= K.mean(inputs, axis=1, keepdims=True)
        inputs = K.l2_normalize(inputs, axis=1)
        pos = K.relu(inputs)
        neg = K.relu(-inputs)
        return K.concatenate([pos, neg], axis=1)

# global parameters
batch_size = 128
num_classes = 10
epochs = 40

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# build the model
model = Sequential()
model.add(layers.Dense(256, input_shape=(784,)))
model.add(Antirectifier())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(256))
model.add(Antirectifier())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# train the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# next, compare with an equivalent network
# with2x bigger Dense layers and ReLU

print("Time taken:", datetime.now() - startTime)
print("\n" * 5)