from matplotlib.animation import FuncAnimation
from scipy import linalg
import matplotlib.pyplot as plt
import numpy as np
import cmath
import math

def gaussian(xs, x, p, d):
    c = (1/(2*math.pi*(d**2)))**.25
    return np.array([c*cmath.exp(-1j*p*i - ((i-x)**2)/(4*(d**2))) for i in xs])

def hamiltonian(n, dx, v = None):
    if v is None:
        v = np.zeros((n, n))
    
    mat = np.zeros((n, n))
    np.fill_diagonal(mat, -2)
    mat[range(1, n), range(n-1)] = 1
    mat[range(n-1), range(1, n)] = 1
    mat /= -2 * dx ** 2

    return mat + v

def timeEvolutionOperator(mat, dt):
    return linalg.expm(-1j * mat * dt)

nPoints = 1000
xStart = -100
xEnd = 100
deltaT = 1
deltaX = (xEnd - xStart) / nPoints
xPoints = np.array([deltaX * i + xStart for i in range(nPoints)])


omega = 1/32

#v = np.array([0 for i in xPoints]) #No Potential
#v = np.array([.5 * omega ** 2 * x ** 2 for x in xPoints])   #Oscillator
#v = np.array([(1 if i >= 95 and i <= 105 else 0) for i in xPoints])
#v = np.array([1-2*abs(x/xEnd) for x in xPoints])   #Pyramid
#v = -np.array([1-2*abs(x/xEnd) for x in xPoints])   #InversePyramid
#v = np.array([(-.95 if x <= 0 else 2 / ((x/xEnd+1))**2) - 1 for x in xPoints])   #Alpha
v = np.array([(-.9 if (abs(x) < 15) else .9) for x in xPoints]) #Well
vmat = np.zeros((nPoints, nPoints))
vmat[range(nPoints), range(nPoints)] = v

xPos = 0
spread = 5
momentum = 0

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5,5)

h = timeEvolutionOperator(hamiltonian(nPoints, deltaX, vmat), deltaT)
g = gaussian(xPoints, xPos, -momentum, spread)

scale = 5
partScale = .5

def animate(i):
    psi = g.copy()
    for j in range(i):
        psi = h @ psi
    ax.clear()
    yPoints = psi.real ** 2 + psi.imag ** 2
    ax.plot(xPoints, yPoints * scale, color='blue')
    ax.plot(xPoints, psi.real * partScale, color='green', lw=.75)
    ax.plot(xPoints, psi.imag * partScale, color='red', lw=.75)
    ax.plot(xPoints, v, color='black', lw = .1)
    ax.set_xlim([xStart, xEnd])
    ax.set_ylim([-1, 1])

ani = FuncAnimation(fig, animate, frames=400, repeat=False)
plt.close()

from matplotlib.animation import PillowWriter

ani.save("test.gif", dpi=300,
         writer=PillowWriter(fps=20))