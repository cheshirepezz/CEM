#
# Created:        01.08.2020
# Last modified:  06.08.2020
# @author:        Luca Pezzini
# e-mail :        luca.pezzini@edu.unito.it
#

#
#                2D CARTESIAN TE_z MODE ON YEE MESH (FDTD)
# Solve the Transverse Magnetic mode equation on the xy-plane (TM_z)
# using the Finite Difference Time Domain metod (FDTD) on the Yee lattice
# in simple cartesian geometry. 
#           Yee Grid (Staggered grid)
#           E[n-1] H[n-1/2] E[n] H[n+1/2] E[n+1]
#           E[0          1          2         3]
#           B[      0          1          2    ]
# The E fields are defined at every unitary space indices (i, j); instead
# B field is defined at fractional indices (i+0.5, j+0.5) as the Yee
# It's used the convention to shift i,j+1/2->i,j+1 & n+1/2->n solving the 
# problem of representing fractional indices.
# Physical parameter are converted in undimensional code units.
# The time update is done using Leapfrog time-stepping. Here, E-fields
# i.e Ex and Ey are updated every full time-step and Bz field is updated every 
# half time-step. This is shown by three alternating for-loop updates in groups 
# of two (for E) and one (for B) spanning entire spatial grid inside a main 
# for-loop for time update spanning the entire time-grid.
#

import math
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
# STEP 1: SET PARAMETER & BC!
#

# Lattice param.
maxITER = 100
x_min, x_max = 0, 1
y_min, y_max = 0, 1
nx, ny = 50, 50  # nodes
Lx = int(abs(x_max-x_min))
Ly = int(abs(y_max-y_min))
deltax = Lx/(nx-1)  # faces are (nodes - 1)
deltay = Ly/(ny-1)
deltat = 1/(100*math.sqrt(2)) # CFL condition

print("LATTICE PARAMETERS!")
print("Numbers of Iterations:", maxITER)
print("Time Step:", deltat)
print("Domain Length (x, y):", Lx, Ly)
print("Number of Nodes (x, y):", nx, ny)
print("Increments (x, y):", deltax, deltay)

#
# STEP 2: DEF. MESH GRID & PHYSICAL PARAM.!
# giving the string ‘ij’ returns a meshgrid with matrix
# indexing, while ‘xy’ returns a meshgrid with Cartesian indexing
#

x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
xv, yv = np.meshgrid(x, y, indexing='ij')

# Initialization of field matrices
Ez = np.zeros([nx, ny])
Bx = np.zeros([nx, ny])
By = np.zeros([nx, ny])
Ez1 = np.zeros([nx, ny])
Bx1 = np.zeros([nx, ny])
By1 = np.zeros([nx, ny])

# Initial conditions
Ez[int((nx-1)/2),int((ny-1)/2)] = 0.001

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

print("START SYSTEM EVOLUTION!")
for time in range(maxITER-1):
    
    # BEGIN : spatial update loops for Bz fields
    #for i in range(1, nx-1):
        #for j in range(1, ny-1):
            #Bz1[i+1, j+1] = Bz[i+1, j+1]+deltat*((1/deltay)*(Ex[i+1, j+1]-Ex[i+1, j])-(1/deltax)*(Ey[i+1, j+1]+Ey[i, j+1]))
    # NOTE: B field is calculate with the updated E field so is half a step ahead
    Ez1[xs+1:xe+1, ys+1:ye+1] = Ez[xs+1:xe+1, ys+1:ye+1]+deltat*((1/deltay)*\
    (Bx[xs+1:xe+1, ys+1:ye+1]-Bx[xs+1:xe+1, ys:ye])-(1/deltax)*\
    (By[xs+1:xe+1, ys+1:ye+1]+By[xs:xe, ys+1:ye+1]))
    # END : spatial update loops for Bz fields
    
    # Reflective BC for B field
    Ez1[xs,ys:ye]=Ez1[xe,ys:ye] #left-right
    Ez1[xs:xe,ys]=Ez1[xs:xe,ye] #top-bottom
    
    Ez = Ez1

    # BEGIN : spatial update loops for Ey and Ex fields
    #for i in range(1, nx-1):
        #for j in range(1, ny-1):
            #Ex1[i+1, j] = Ex[i+1, j]+(deltat/deltay)*(Bz[i+1, j+1]-Bz[i+1, j])
            #Ey1[i, j+1] = Ey[i, j+1]-(deltat/deltax)*(Bz[i+1, j+1]-Bz[i, j+1])
    Bx1[xs+1:xe+1, ys:ye] = Bx[xs+1:xe+1, ys:ye]+(deltat/deltay)*\
    (Ez[xs+1:xe+1, ys+1:ye+1]-Ez[xs+1:xe+1, ys:ye])
    By1[xs:xe, ys+1:ye+1] = By[xs:xe, ys+1:ye+1]-(deltat/deltax)*\
    (Ez[xs+1:xe+1, ys+1:ye+1]-Ez[xs:xe, ys+1:ye+1])
    # END : spatial update loops for Ey and Ex fields

    # Reflective BC for E field
    Bx1[xs,ys:ye]=Bx1[xe,ys:ye] #left-right
    Bx1[xs:xe,ys]=Bx1[xs:xe,ye] #top-bottom
    By1[xs,ys:ye]=Bx1[xe,ys:ye] #left-right
    By1[xs:xe,ys]=Bx1[xs:xe,ye] #top-bottom
    
    # Update E field
    Bx = Bx1
    By = By1

    if flag_plt == False:
        plt.figure(figsize =(12, 10))

    plt.pcolor(xv, yv, Ez1)
    plt.title("Map of Ez")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.colorbar()

    if flag_start == False:
        command = input("Press Enter, then start.")
        flag_start = True
        
    plt.pause(0.001)
    plt.clf()
    flag_plt=True

print("DONE!")

#
# STEP 4: VISUALIZATION!
#

plot2D(xv, yv, Ez1)

plt.figure(3)
plt.contourf(xv, yv, Ez1, cmap=plt.cm.jet)
plt.title("Contour of Ez")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()

#plt.figure(2)
#plt.pcolor(xv, yv, Ez1)
#plt.title("Map of Ez")
#plt.xlabel("y")
#plt.ylabel("x")
#plt.colorbar()

plt.show()


