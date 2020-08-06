#
# Created:        01.08.2020
# Last modified:  06.08.2020
# @author:        Luca Pezzini
# e-mail :        luca.pezzini@edu.unito.it
#

#
#                2D CARTESIAN TE_z MODE ON YEE MESH (FDTD)
# Solve the Transverse Electric mode equation on the xy-plane (TE_z)
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
Bz = np.zeros([nx, ny])
Ex = np.zeros([nx, ny])
Ey = np.zeros([nx, ny])
Bz1 = np.zeros([nx, ny])
Ex1 = np.zeros([nx, ny])
Ey1 = np.zeros([nx, ny])

# Initial conditions
Bz[int((nx-1)/2),int((ny-1)/2)] = 0.001

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
    
    # BEGIN : spatial update loops for Ey and Ex fields
    #for i in range(1, nx-1):
        #for j in range(1, ny-1):
            #Ex1[i+1, j] = Ex[i+1, j]+(deltat/deltay)*(Bz[i+1, j+1]-Bz[i+1, j])
            #Ey1[i, j+1] = Ey[i, j+1]-(deltat/deltax)*(Bz[i+1, j+1]-Bz[i, j+1])
    Ex1[xs+1:xe+1, ys:ye] = Ex[xs+1:xe+1, ys:ye]+(deltat/deltay)*\
    (Bz[xs+1:xe+1, ys+1:ye+1]-Bz[xs+1:xe+1, ys:ye])
    Ey1[xs:xe, ys+1:ye+1] = Ey[xs:xe, ys+1:ye+1]-(deltat/deltax)*\
    (Bz[xs+1:xe+1, ys+1:ye+1]-Bz[xs:xe, ys+1:ye+1])
    # END : spatial update loops for Ey and Ex fields

    # Reflective BC for E field
    Ex1[xs,ys:ye]=Ex1[xe,ys:ye] #left-right
    Ex1[xs:xe,ys]=Ex1[xs:xe,ye] #top-bottom
    Ey1[xs,ys:ye]=Ex1[xe,ys:ye] #left-right
    Ey1[xs:xe,ys]=Ex1[xs:xe,ye] #top-bottom
    
    # Update E field
    Ex = Ex1
    Ey = Ey1

    # BEGIN : spatial update loops for Bz fields
    #for i in range(1, nx-1):
        #for j in range(1, ny-1):
            #Bz1[i+1, j+1] = Bz[i+1, j+1]+deltat*((1/deltay)*(Ex[i+1, j+1]-Ex[i+1, j])-(1/deltax)*(Ey[i+1, j+1]+Ey[i, j+1]))
    # NOTE: B field is calculate with the updated E field so is half a step ahead
    Bz1[xs+1:xe+1, ys+1:ye+1] = Bz[xs+1:xe+1, ys+1:ye+1]+deltat*((1/deltay)*\
    (Ex[xs+1:xe+1, ys+1:ye+1]-Ex[xs+1:xe+1, ys:ye])-(1/deltax)*\
    (Ey[xs+1:xe+1, ys+1:ye+1]+Ey[xs:xe, ys+1:ye+1]))
    # END : spatial update loops for Bz fields
    
    # Reflective BC for B field
    Bz1[xs,ys:ye]=Bz1[xe,ys:ye] #left-right
    Bz1[xs:xe,ys]=Bz1[xs:xe,ye] #top-bottom
    
    Bz = Bz1

    if flag_plt == False:
        plt.figure(figsize =(12, 10))

    plt.pcolor(xv, yv, Bz1)
    plt.title("Map of Bz")
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

plot2D(xv, yv, Bz1)

plt.figure(3)
plt.contourf(xv, yv, Bz1, cmap=plt.cm.jet)
plt.title("Contour of Bz")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()

#plt.figure(2)
#plt.pcolor(xv, yv, Bz1)
#plt.title("Map of Bz")
#plt.xlabel("y")
#plt.ylabel("x")
#plt.colorbar()

plt.show()

