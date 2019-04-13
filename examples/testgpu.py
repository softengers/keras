import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
startTime = datetime.now()

"""
cpu 花了3分钟：

Shape: (1500, 1500) Device: /cpu:0
Time taken: 0:03:09.855242

Shape: (4500, 4500) Device: /cpu:0
Time taken: 0:00:02.618007

Shape: (7500, 7500) Device: /cpu:0
Time taken: 0:00:02.618007

Shape: (9500, 9500) Device: /cpu:0
Time taken: 0:00:10.644696

Shape: (12500, 12500) Device: /cpu:0
Time taken: 0:00:22.554438

Shape: (15000, 15000) Device: /cpu:0
Time taken: 0:00:40.408174

Shape: (20000, 20000) Device: /cpu:0
Time taken: 0:01:20.123051【80】

Shape: (23300, 23300) Device: /cpu:0
Time taken: 0:02:11.564943【131，131/4.7=28倍】
gup 花了4秒钟：

Shape: (1500, 1500) Device: /gpu:0
Time taken: 0:00:04.823442

Shape: (4500, 4500) Device: /gpu:0
Time taken: 0:00:02.529637
 
Shape: (7500, 7500) Device: /gpu:0
Time taken: 0:00:03.234047

Shape: (9500, 9500) Device: /gpu:0
Time taken: 0:00:04.673639

Shape: (11000, 11000) Device: /gpu:0
Time taken: 0:00:04.710180

Shape: (12500, 12500) Device: /gpu:0
Time taken: 0:00:04.290680

Shape: (12500, 12500) Device: /gpu:0 【RTX2070】
Time taken: 0:00:02.022930

Shape: (15000, 15000) Device: /gpu:0  【RTX2070】
Time taken: 0:00:02.307800

Shape: (20000, 20000) Device: /gpu:0 【RTX2070，3s，CPU，80，80/3=26 CPU比GPU满30倍，谷歌的专用机器学习芯片TPU处理速度要比GPU和CPU30倍】
Time taken: 0:00:03.575326

Shape: (21000, 21000) Device: /gpu:0
Time taken: 0:00:03.748350

Shape: (22000, 22000) Device: /gpu:0
Time taken: 0:00:09.433556

Shape: (23000, 23000) Device: /gpu:0
Time taken: 0:00:15.026813

Shape: (23200, 23200) Device: /gpu:0
Time taken: 0:00:04.657253

Shape: (23300, 23300) Device: /gpu:0
Time taken: 0:00:04.718171

24000：tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor with shape[24000,24000] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
	 [[node MatMul (defined at C:/Users/admin/PycharmProjects/keras/examples/testgpu.py:76) ]]

用GPU做DeepLearning要比CPU快40～80倍

The speed difference of CPU and GPU can be significant in deep learning. But how much? Let’s do a test.

The computer:

The computer I use is a Amazon AWS instance g2.2xlarge (https://aws.amazon.com/ec2/instance-types/). The cost is 0.65/hour,or

15.6/day, or $468/mo. It has one GPU (High-performance NVIDIA GPUs, each with 1,536 CUDA cores and 4GB of video memory), and 8 vCPU (High Frequency Intel Xeon E5-2670 (Sandy Bridge) Processors). Memory is 15G.

原文链接by Xu Cui

"""
sysarg1 = 'cpu'
sysarg2 = 23300
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
