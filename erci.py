from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    tf.summary.histogram("weight", Weights)
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    tf.summary.histogram("biases", biases);
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    tf.summary.histogram("Wx_plus_b", Wx_plus_b)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram("outputs", outputs)
    return outputs
'''
# Make up some real data
x_data = np.linspace(-1,1,10)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

'''
x_data = np.asarray([-14, -13, -9, 5, -5, -3, -2, 3, -8, 8, -7, -4.0, -1.0, 0, 2.0, 6.0, 7.0, -6.0, -10.0, 9.0, 15.0])[:, np.newaxis]
y_data = np.asarray([196, 169, 81, 25, 25, 9, 4, 9, 64, 64, 49, 17.0, 1.0, 0, 4.0, 37.0, 48.0, 36.0, 99.0, 80.0, 228.0])[:, np.newaxis]

x_data = np.asarray([ -0.8, 0.8, -0.7, -0.4, -1.0, 0, 0.2, 0.6, 0.7, -0.6, -0.1, 0.9, 0.15])[:, np.newaxis]
y_data = np.asarray([0.64, 0.64, 0.49, 0.17, 1.0, 0, 0.04, 0.37, 0.48, 0.36, 0.01, 0.8, 0.0228])[:, np.newaxis]

x_data = np.asarray([ -1.0,  -0.8, -0.7, -0.6, -0.4, -0.1, 0, 0.15 , 0.2, 0.6, 0.7, 0.8,0.9])[:, np.newaxis]
y_data = np.asarray([ 1.0, 0.64,  0.49, 0.36, 0.17, 0.01, 0, 0.0228,  0.04, 0.37, 0.48, 0.64,0.8])[:, np.newaxis]

#x_data = np.asarray([-4, 0, 4])[:, np.newaxis]
#y_data = np.asarray([16, 0, 16])[:, np.newaxis]

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
tf.summary.scalar("loss", loss)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
# train_step = tf.train.AdamOptimizer(0.05).minimize(loss)

# important step
init = tf.initialize_all_variables()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("desktop/", sess.graph)

sess.run(init)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#ax.axis([-20,20,-300,300])
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(10000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)
