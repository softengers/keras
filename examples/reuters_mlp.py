'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.

tensorflow.python.framework.errors_impl.InternalError: Blas GEMM launch failed : a.shape=(32, 1000), b.shape=(1000, 512), m=32, n=512, k=1000
	 [[{{node dense_1/MatMul}}]]

8083/8083 [==============================] - 1s 113us/step - loss: 0.4170 - acc: 0.9004 - val_loss: 0.8595 - val_acc: 0.8154
Epoch 5/5
	 8083/8083 [==============================] - 1s 103us/step - loss: 0.3298 - acc: 0.9191 - val_loss: 0.8736 - val_acc: 0.8198
  32/2246 [..............................] - ETA: 0s
1408/2246 [=================>............] - ETA: 0s
2246/2246 [==============================] - 0s 36us/step
Test score: 0.8689914898265183
Test accuracy: 0.7969723954226221
Time taken: 0:00:08.677153

8083/8083 [==============================] - 1s 103us/step - loss: 0.1021 - acc: 0.9623 - val_loss: 1.5924 - val_acc: 0.7942
Epoch 100/100
8083/8083 [==============================] - 1s 102us/step - loss: 0.0999 - acc: 0.9644 - val_loss: 1.6652 - val_acc: 0.7909
  32/2246 [..............................] - ETA: 0s
1600/2246 [====================>.........] - ETA: 0s
2246/2246 [==============================] - 0s 33us/step
Test score: 1.7010244841248672
Test accuracy: 0.782279608192342
Time taken: 0:01:31.067147
'''
from __future__ import print_function

import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

from datetime import datetime
startTime = datetime.now()

max_words = 1000
batch_size = 32
epochs = 100

print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print("Time taken:", datetime.now() - startTime)
print("\n" * 5)