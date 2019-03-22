import tensorflow as tf
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

#samples = random.sample(range(60), 30)
#print samples

arr = rnd.random(size=70)*12
avg = np.mean(arr)
x_data = arr - avg

x = np.linspace(-1, 1, 70)
noise = rnd.normal(0, 10, x.shape)
y_data = 2*x_data**3+2*x_data**2-40*x_data+3 + noise
#y_data = 2*x_data**3+40*x_data+3+noise
#y_data = 2*x_data**2+40*x_data+3+noise

'''
fig = plt.figure()
axes1 = fig.add_subplot(111)
axes1.plot(x_data, y_data, 'ro')
plt.show()
'''

xs = tf.placeholder(tf.float32)
ys = tf.placeholder(tf.float32)

#a*x^3+b*x^2+c*x+d

a = tf.Variable(rnd.randn())
b = tf.Variable(rnd.randn())
c = tf.Variable(rnd.randn())
d = tf.Variable(rnd.randn())

activation = tf.add(tf.add(tf.add(tf.multiply(a, tf.pow(xs, 3)), tf.multiply(b, tf.pow(xs, 2))), tf.multiply(c, xs)), d)
#activation = tf.add(tf.add(tf.multiply(a, tf.pow(xs, 3)), tf.multiply(c, xs)), d)
#activation = tf.add(tf.add(tf.multiply(a, tf.pow(xs, 2)), tf.multiply(b, xs)), c)
learnRate = 0.1
cost = tf.reduce_sum(tf.pow(activation-ys, 2))/(140)
optimizer = tf.train.AdamOptimizer(0.05).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init)
'''
opt = a.assign(1)
sess.run(opt)
opt = b.assign(2)
sess.run(opt)
opt = c.assign(-11)
sess.run(opt)
opt = d.assign(1)
sess.run(opt)
print("a = " + str(sess.run(a)) + ", b = " + str(sess.run(b)) + ", c = " + str(sess.run(c)) + ", d = " + str(sess.run(d)))
'''
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
ax.axis([-10, 10, -400, 400])
plt.ion()
plt.show()

for i in range(1000):
	for (x1, y1) in zip(x_data, y_data):
		sess.run(optimizer, feed_dict={xs:x1, ys:y1})
		#optimizer = tf.train.GradientDescentOptimizer(learnRate).minimize(cost)
	if(i%2==0):
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
		print("cost = " + str(sess.run(cost, feed_dict={xs:x_data, ys:y_data})))
		lines = ax.plot(x*8, sess.run(activation, feed_dict={xs: x*8}), 'r-', lw=3)
		plt.pause(0.02)
		print("a = " + str(sess.run(a)) + ", b = " + str(sess.run(b)) + ", c = " + str(sess.run(c)) + ", d = " + str(sess.run(d)))


'''
plt.plot(x, y, 'ro')
plt.show()
'''
