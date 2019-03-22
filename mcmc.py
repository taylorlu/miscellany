import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import time
import math
import matplotlib
from matplotlib.animation import FuncAnimation
'''
left = [0,1,2,3,4,5]
height = [1,5,3,9,4,7]
plt.bar(left,height,color='r')
plt.show()

c = 0
for i in range(0, 100):
	height[c%6] = i%20+height[c%6]
	plt.set_height(height)
	plt.bar(left,height,color='r')
	plt.show()
	time.sleep(0.1)
'''
'''
xData = [0,1,2,3,4,5,6,7]
yData = [3,4,2,5,7,4,6,1]
xData = np.linspace(0,8,100)
yData = np.sin(xData)
plt.plot(xData, yData)


fig = plt.figure()
ax = fig.add_subplot(1,1,1,xlim=(0,8),ylim=(-4,4))
line, = ax.plot([],[])

def animate(i):
	x = np.linspace(0,8,1000)
	y = np.sin(2*np.pi*i*x)
	line.set_data(x, y)
	return line

ani = animation.FuncAnimation(fig, animate, frames = 10, interval =100)
plt.show()

'''

sample_count = 100
meanValue = sample_count/2
xData = np.linspace(1,sample_count,sample_count)
#height = np.random.random(size=sample_count)*10
height = np.zeros(sample_count)

initX = 10
fig = plt.figure()
rects = plt.bar(xData, height, color='g')
ax = plt.gca()
ax.set_yticks(np.linspace(0,100,10))


def normal_distribution_func(x):
    return 0.008*math.exp(-((x-meanValue)**2)/meanValue) + 0.000001

burninCount = 100
cycleCount = 0

def mcmc(inputs):

    global initX
    global cycleCount
    cycleCount += 1
    cur_prob = normal_distribution_func(initX)
    random_x = int(sample_count * np.random.random(size=1))
   # print "random = %d" % random_x
    random_prob = normal_distribution_func(random_x)
    if(cur_prob > random_prob):
        jumpRate = float(random_prob)/float(cur_prob)
        randRate = np.random.random(size=1)
#        print "jumpRate = %f , randRate = %f" % (jumpRate,randRate)
        if(randRate<jumpRate):
            if(cycleCount>burninCount):
                inputs[random_x] += 1
            initX = random_x
        else:
            mcmc(inputs)
    else:
        if(cycleCount>burninCount):
            inputs[random_x] += 1
        initX = random_x

def animate(frame):

#    for i in range(0, len(xData)):
#        height[i] = 5*np.fabs(np.sin(np.random.random(size=1)))
    mcmc(height)
    for rect, h in zip(rects, height):
        rect.set_height(h)
    fig.canvas.draw()

ani = animation.FuncAnimation(fig, animate, frames = 10, interval =1)
plt.show()
'''

# Gibbs Sampling

mean = [50.0, 50.0]
cov = [[40.0, 20.0], [10.0, 20.0]]
covInv = np.linalg.inv(cov)
sample_count = 100
burninCount = 100
cycleCount = 0
initX = 10.0
initY = 10.0
turnToX = True
initCoord = (initX, initY)
random_xy = (initX, initY)

plt.ion()
fig, ax = plt.subplots()
scatterPlot = plt.scatter([], [], s = 1)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

def multi_normal_distribution_func(x, y):
    center = np.array([x, y]) - mean
    ret = np.exp(-np.dot(np.dot(center.T, covInv), center)/2)/(6.28*np.sqrt(np.linalg.det(cov)))
    return ret

def gibbSampling():

    global initX
    global initY
    global cycleCount
    global initCoord
    global turnToX
    cycleCount += 1

    cur_prob = multi_normal_distribution_func(initCoord[0], initCoord[1])

    #print "random = %d" % random_x

    if(turnToX):
        random_xy = (sample_count * np.random.random(size=1)[0], initY)
    else:
        random_xy = (initX, sample_count * np.random.random(size=1)[0])

    random_prob = multi_normal_distribution_func(random_xy[0], random_xy[1])

 #   print "%f, %f\r\n" % (cur_prob, random_prob)
    if(cur_prob > random_prob):
        jumpRate = float(random_prob)/float(cur_prob)
        randRate = np.random.random(size=1)[0]
#        print "jumpRate = %f , randRate = %f" % (jumpRate,randRate)
        if(randRate<jumpRate):
#            if(cycleCount>burninCount):
            initCoord = random_xy
            if(turnToX):
                initX = random_xy[0]
                turnToX = False
            else:
                initY = random_xy[1]
                turnToX = True
    else:
#        if(cycleCount>burninCount):
        initCoord = random_xy
        if(turnToX):
            initX = random_xy[0]
            turnToX = False
        else:
            initY = random_xy[1]
            turnToX = True

while True:
    gibbSampling()
    array = scatterPlot.get_offsets()
    array = np.append(array, (initCoord[0], initCoord[1]))
    scatterPlot.set_offsets(array)
    fig.canvas.draw()
'''