import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

"""
cpu 花了3分钟：

Shape: (1500, 1500) Device: /cpu:0
Time taken: 0:03:09.855242

Shape: (4500, 4500) Device: /cpu:0
Time taken: 0:00:02.618007

Shape: (15000, 15000) Device: /cpu:0
Time taken: 0:00:40.408174


gup 花了4秒钟：

Shape: (1500, 1500) Device: /gpu:0
Time taken: 0:00:04.823442

Shape: (4500, 4500) Device: /gpu:0
 

"""
sysarg1 = 'gpu'
sysarg2 = 4500
device_name = sysarg1 # Choose device from cmd line. Options: gpu or cpu
shape = (sysarg2, sysarg2)

if device_name == 'cpu':
    device_name = "/cpu:0"
else:
    device_name = "/gpu:0"

with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)

startTime = datetime.now()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    result = session.run(sum_operation)
    print(result)

# It can be hard to see the results on the terminal with lots of output --
# add some newlines to improve readability.
#
print("\n" * 5)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)
print("\n" * 5)
