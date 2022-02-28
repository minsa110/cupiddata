# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 1000)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

# %%
# View and edit nested variables
a = [1, 2, 3]
b = [[1, 2, 3, ['A', 'B', 'C']], 'a', 'b', 'c']

# %%
m = 1j
n = complex(3,4)

# %%
d = dict(
    Colorado='Rockies',
    Boston='Red Sox',
    Minnesota='Twins',
    Milwaukee='Brewers',
    Seattle='Mariners',
    nums=[1,2,3,4,5],
    scale=0.001
)

# %%
# Custom class/object
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

p1 = Person("John", 36)

# %%

c = [Person("Sam", 22), ['A', 'B', 'C']]

# %%
alpha = 0.7
phi_ext = 2 * np.pi * 0.5

def flux_qubit_potential(phi_m, phi_p):
    return 2 + alpha - 2 * np.cos(phi_p) * np.cos(phi_m) - alpha * np.cos(phi_ext - 2*phi_p)

phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)
X,Y = np.meshgrid(phi_p, phi_m)
Z = flux_qubit_potential(X, Y).T

# %%
from mpl_toolkits.mplot3d.axes3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)

# %%
import datetime as dt
import matplotlib.animation as animation

anifig = plt.figure()
aniax = anifig.add_subplot(1,1,1)
xs = []
ys = []

def animate(i, xs, ys):
    temp_c = round(np.random.random(), 2)
    
    xs.append(dt.datetime.now().strftime("%H:%M:%S.%f"))
    ys.append(temp_c)
    
    xs = xs[-20:]
    ys = ys[-20:]
    
    aniax.clear()
    aniax.plot(xs, ys)
    
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title("Temperature Data")
    plt.ylabel("Temperature (deg C)")
    
ani = animation.FuncAnimation(anifig, animate, fargs=(xs, ys), interval=100)
plt.show()