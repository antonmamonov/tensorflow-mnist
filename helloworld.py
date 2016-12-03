import tensorflow as tf
import numpy as np

inputs = 2
inputData = []
wData = []
b = tf.Variable(tf.zeros([1]))

actualValues = [1,3]
actualBias = 4

y_data = np.zeros(100)

Ws = []

for i in range(inputs):
  inputX = np.random.rand(100).astype(np.float32)
  inputData.append(inputX)
  wData.append(tf.Variable(tf.random_uniform([1], -1.0, 1.0)))
  y_data += inputX * actualValues[i]
  Ws.append(tf.Variable(tf.random_uniform([1], -1.0, 1.0)))

y_data += actualBias

b = tf.Variable(tf.zeros([1]))
y = tf.mul(inputData,Ws) + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

tf.scalar_summary("loss",loss)
tf.scalar_summary(['b'],b)
tf.scalar_summary(['W1','W2'],W)

mergedSummaryOp = tf.merge_all_summaries()
summaryWriter = tf.train.SummaryWriter("/code/events")

# print mergedSummaryOp
# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
    sess.run(train)
    summary = sess.run(mergedSummaryOp)
    summaryWriter.add_summary(summary,step)

Learns best fit is W: [0.1], b: [0.3]