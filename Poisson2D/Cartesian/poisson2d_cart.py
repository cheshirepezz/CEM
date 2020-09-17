#
# Created:        02.08.2020
# Last modified:  06.08.2020
# @author:        Luca Pezzini
# e-mail :        luca.pezzini@edu.unito.it
#

#
#              2D CARTESIAN POISSON EQ. ON YEE MESH FDTD
# Solve the Poisson eq. on the xy-plane in simple Cartesian geometry
# using the Finite Difference Time Domain metod (FDTD) on the Yee lattice. 
# It's used the convention to shift i,j+1/2->i,j+1 & n+1/2->n 
# solving the problem of representing fractional indices.
# Physical parameter are converted in undimensional code units.
# The time update is done using Leapfrog time-stepping.
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

def plot2D(x, y, p):
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, p[:], rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False)
    ax.view_init(30, 225)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.show()

#
# STEP 1: SET PARAMETER!
#

# Lattice param.
maxITER = 50
x_min, x_max = 0, 1
y_min, y_max = 0, 1
nx, ny = 50, 50  # nodes
Lx = int(abs(x_max-x_min))
Ly = int(abs(y_max-y_min))
deltax = Lx/(nx-1)  # faces are (nodes - 1)
deltay = Ly/(ny-1)

print("LATTICE PARAMETERS!")
print("Numbers of Iterations:", maxITER)
print("Domain Length (x, y):", Lx, Ly)
print("Number of Nodes (x, y):", nx, ny)
print("Increments (x, y):", deltax, deltay)

#
# STEP 2: DEF. MESH GRID & PHYSICAL PARAM.!
# giving the string ‘ij’ returns a meshgrid with matrix
# indexing, while ‘xy’ returns a meshgrid with Cartesian indexing
#

x = np.linspace(x_min, x_max, nx)
y = np.linspace(x_min, x_max, ny)
xv, yv = np.meshgrid(x, y, indexing='ij')

# Initialization of field matrices
poisson = np.zeros((ny, nx))
phy = np.zeros((ny, nx))
s = np.zeros((ny, nx)) # source
##s[int(3 * ny / 4), int(3 * nx / 4)] = -100

#
# BOUNDARY CONDITIONS
#

bc_left, bc_right = 0, 0
bc_top, bc_bottom = 0, 10

print("BOUNDARY CONDITIONS!")
print("Left:", bc_left)
print("Right:", bc_right)
print("Top:", bc_top)
print("Bottom:", bc_bottom)

phy[0:(nx-1), 0] = bc_bottom
phy[0:(ny-1), (ny-1)] = bc_top
phy[0, 0:(ny-1)] = bc_left
phy[(nx-1), 0:(ny-1)] = bc_right

#
# STEP 3: TIME EVOLUTION OF THE FIELD ON THE GRID!
#

# Start & End
xs = 1
xe = nx-1
ys = 1
ye = ny-1

# Flag
flag_plt = False
flag_start = False

# BEGIN : time update loops for phy field
for time in range(maxITER):

    # BEGIN : spatial update loops for phy field
    #for i in range(1, nx-1):
        #for j in range(1, ny-1):
            #poisson[i, j] = 1/(deltax*deltax)*(phy[i+1,j]-2*phy[i,j]+pd[i-1,j])+1/(deltay*deltay)*(phy[i,j+1]-2*phy[i,j]+phy[i,j-1])-s[i,j]
    
    poisson[xs:xe, ys:ye] = 1/(deltax*deltax)*(phy[xs+1:xe+1, ys:ye]-\
    2*phy[xs:xe, ys:ye]+phy[xs-1:xe-1, ys:ye])+1/(deltay*deltay)*\
    (phy[xs:xe, ys+1:ye+1]-2*phy[xs:xe, ys:ye]+phy[xs:xe, ys-1:ye-1])-\
    s[xs:xe, ys:ye]
    
    # END : spatial update loops for phy field
    
    phy = poisson
    if flag_plt == False:
        plt.figure(figsize =(12, 10))

    plt.title("Contour of phy")
    #plt.contourf(xv, yv, poisson, cmap=plt.cm.jet)
    plt.pcolor(xv, yv, poisson)
    plt.colorbar()

    if flag_start == False:
        command = input("Press Enter, then start.")
        flag_start = True
        
    plt.pause(0.001)
    plt.clf()
    flag_plt=True
    
# END : time update loops for phy field

#
# STEP 4: VISUALIZATION!
#

plot2D(xv, yv, poisson)
'''
plt.title("Contour of phy")
#plt.contourf(xv, yv, poisson, cmap=plt.cm.jet)
plt.pcolor(xv, yv, poisson)
plt.colorbar()
'''
plt.show()







