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
n_iterations = 1000
learning_rate = 0.0005

# input X: 
X = tf.placeholder(tf.float32, [None,32*32])
# 
Y_ = tf.placeholder(tf.float32, [None, 10])
# weights W[1024, 10]   
W = tf.Variable(tf.zeros([1024, 10]))
# biases b[10]
B = tf.Variable(tf.zeros([10]))


# model
Y = tf.nn.softmax(tf.matmul(X,W) + B)

# loss function
cross_entropy = -tf.reduce_sum(Y_*tf.log(Y))

# accuracy
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
accuracies = []

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

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
	
