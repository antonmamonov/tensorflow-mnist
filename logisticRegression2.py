%matplotlib inline
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

XData = np.array([
        [1,4],
        [5,7],
        [2,5],
        [3,6],
        
        [6,1],
        [7,2],
        [8,3.5],
        [9,4.25]
    ])

YData = np.array([
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        
        [0,1],
        [0,1],
        [0,1],
        [0,1]
    ])


reds = {
    'x':[],
    'y':[]
}
for n,d in enumerate(YData):
    if d[0] == 1:
        reds['x'].append(XData[n][0])
        reds['y'].append(XData[n][1])

blues = {
    'x':[],
    'y':[]
}
for n,d in enumerate(YData):
    if d[1] == 1:
        blues['x'].append(XData[n][0])
        blues['y'].append(XData[n][1])

plt.plot(blues['x'], blues['y'], 'bo')
plt.plot(reds['x'], reds['y'], 'ro')
plt.show()

learningRate = 0.1
epoch = 100

X = tf.placeholder("float", [None, 2])
Y = tf.placeholder("float", [None, 2])

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

W = init_weights([1, 2])
B = init_weights([1, 1])
hyp = tf.nn.sigmoid(tf.add(tf.mul(X,W),B))
loss = tf.reduce_sum(tf.square(hyp-Y))
opt = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

losses = []

for epoch_i in range(epoch):
    for n,t in enumerate(XData):
        sess.run(opt,feed_dict={X:[XData[n]],Y:[YData[n]]})
    
    lossRate = sess.run(loss,feed_dict={X: XData,Y:YData})
    losses.append(lossRate)
    if epoch_i % 10 == 0:
        print(lossRate,sess.run(W),sess.run(B))

plt.plot(losses, range(epoch), 'b',label='loss')
plt.legend()
plt.show()


# print sess.run(W)
# print sess.run(B)
# print sess.run(loss,feed_dict={X: [[1,4],[4,1]],Y:[[1,0],[0,1]]})