
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation

SIZE = 40
STEPS = 10000
colorMap =np.array(["r","g","b"])

fig = plt.figure()

#----------------------------------------------------------------------#
#   Check periodic boundary conditions 
#----------------------------------------------------------------------#
def bc(i):
    if i+1 > SIZE-1:
        return 0
    if i-1 < 0:
        return SIZE-1
    else:
        return i

#----------------------------------------------------------------------#
#   Calculate internal energy
#----------------------------------------------------------------------#
def energy(system, N, M):
    return -1 * system[N,M] * (system[bc(N-1), M]  + system[bc(N+1), M]  + system[N, bc(M-1)]  + system[N, bc(M+1)])

#----------------------------------------------------------------------#
#   Build the system
#----------------------------------------------------------------------#
def build_system():
    system = np.random.random_integers(0,1,(SIZE,SIZE))
    system[system==0] =- 1
    #print colorMap[system + 1]

    return system

#----------------------------------------------------------------------#
#   The Main monte carlo loop
#----------------------------------------------------------------------#

def animate(frame):

    M = np.random.randint(0,SIZE)
    N = np.random.randint(0,SIZE)

    E = -2. * energy(system, N, M)

    if E <= 0.:
        system[N,M] *= -1
    elif np.exp(-1./T*E) > np.random.rand():
        system[N,M] *= -1

    scatter = plt.scatter(M,N)
    if(system[N,M] == -1):
        scatter.set_color('r')
    else:
        scatter.set_color('b')
    fig.canvas.draw()

#----------------------------------------------------------------------#
#   Run the menu for the monte carlo simulation
#----------------------------------------------------------------------#

T = 5

system = build_system()
map2 = colorMap[system + 1]

x,y = np.argwhere(system == -1).T
plt.scatter(x,y, c='r')
x,y = np.argwhere(system == 1).T
plt.scatter(x,y, c='b')

ani = animation.FuncAnimation(fig, animate, frames = 10000)
# ani.save('folder/basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
