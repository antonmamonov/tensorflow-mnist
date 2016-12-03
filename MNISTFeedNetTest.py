%matplotlib inline
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

imageSize = 28
viewImageIndex = 4

random_image = np.reshape(trX[viewImageIndex],(imageSize,imageSize))

plt.imshow(random_image, cmap='gray', interpolation='nearest');
print trY[viewImageIndex]

numInputs = 28 ** 2
numOutputs = 10
learningRate = 0.001
epoch = 25
NUM_CORES = 4

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
    

X = tf.placeholder("float", [None, numInputs])
Y = tf.placeholder("float", [None, numOutputs])

W_i = init_weights([1, numInputs])
W_o = init_weights([numInputs,numOutputs])

B_i = init_weights([1, 1])
B_o = init_weights([1, 1])

hyp_i = tf.nn.sigmoid(tf.add(tf.mul(X,W_i),B_i))
hyp_o = tf.nn.softmax(tf.nn.sigmoid(tf.add(tf.matmul(hyp_i,W_o),B_o)))

loss = tf.reduce_sum(tf.square(hyp_o-Y))
opt = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

NUM_CORES = 4  # Choose how many cores to use.
sess = tf.Session(
    config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                   intra_op_parallelism_threads=NUM_CORES))

sess.run(tf.initialize_all_variables())

## Get final probability
hyp_o_out = sess.run(hyp_o,feed_dict={X: [trX[0]]})
# print hyp_o_out.shape
# print hyp_o_out

## Test loss 
# print sess.run(tf.reduce_sum(tf.square(hyp_o_out - np.array([trY[0]]))))

## Test loss function
### Test one input
# loss_out = sess.run(loss,feed_dict={X: np.array([trX[0]]),Y : np.array([trY[0]])})
### Test entire set
# loss_out = sess.run(loss,feed_dict={X: trX,Y : trY})
# print loss_out

losses = []

print "Training"
for epoch_i in range(epoch):
    for n,t in enumerate(trX):
        sess.run(opt,feed_dict={X: np.array([trX[n]]),Y : np.array([trY[n]])})
    
    lossRate = sess.run(loss,feed_dict={X: trX,Y:trY})
    losses.append(lossRate)
    print(lossRate)

plt.plot(range(epoch),losses, 'b',label='loss')
plt.legend()
plt.show()