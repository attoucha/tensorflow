import numpy as np
import tensorflow as tf
from util import* 
import matplotlib
import matplotlib.pyplot as plt

tf.set_random_seed(0)

# load data
train_data, test_data = load_data()

batch_size = 100
n_batch = len(train_data[0]) // batch_size
n_iterations = 250
learning_rate = 0.0005

# input X: 
X = tf.placeholder(tf.float32, [None,32*32])
# 
Y_ = tf.placeholder(tf.float32, [None, 10])
# number of neurons single hidden layer
N = 256
# weights W[1024,N] 
W1 = tf.Variable(tf.truncated_normal([1024, N], stddev=0.1))
# biases b[N]
B1 = tf.Variable(tf.ones([N])/10)
# weights W[N,10] 
W2 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
# biases b[10]
B2 = tf.Variable(tf.zeros([10]))

# model
Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Ylogits = tf.matmul(Y1, W2) + B2
Y = tf.nn.softmax(Ylogits)

# loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
accuracies = []

# training step
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(n_iterations):
	train_data = shuffle_data(train_data[0], train_data[1])
	for i in range(n_batch):
		batch_X, batch_Y = train_data[0][i*batch_size:(i+1)*batch_size], train_data[1][i*batch_size:(i+1)*batch_size]
		sess.run(train_step, feed_dict={X:batch_X, Y_:batch_Y})
	test_accuracy = accuracy.eval(session = sess ,feed_dict={X:test_data[0], Y_: test_data[1]})
	accuracies.append(test_accuracy)


plt.plot(accuracies)

plt.show()
	
