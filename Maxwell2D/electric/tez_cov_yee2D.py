#
# created:        03.10.2020
# last modified:  06.10.2020
# author:         Luca Pezzini
# e-mail :        luca.pezzini@edu.unito.it
# MIT license

#
# Please feel free to use and modify this code, but keep the above information. 
# Thanks!
#

#
#                     GR PiC 2D on Yee Lattice
# Solve the Transverse Electric mode equation on the xy-plane (TE_z)
# using the Finite Difference Time Domain metod (FDTD) in general geometry. 
#           Yee Grid (Staggered grid)
#           E[n-1] H[n-1/2] E[n] H[n+1/2] E[n+1]
#           E[0          1          2          3]
#           B[      0          1          2     ]
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
from mpl_toolkits.mplot3d import Axes3D

# Function def.

def Energy(p):
    plt.figure()
    plt.plot(p)
    plt.title("Energy Conservation")
    plt.xlabel("time")
    plt.ylabel("U")

def DivE(p):
    plt.figure()
    plt.plot(p)
    plt.title("Mimetic Operator")
    plt.xlabel("time")
    plt.ylabel("div(E)")

def Plot2D(x, y, p):
    plt.figure()
    plt.contourf(x, y, p)#, cmap=plt.cm.jet)
    plt.title("Contour of Bz")
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.colorbar()

def Plot3D(x, y, p):
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, p[:], rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False)
    ax.view_init(30, 225)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.show()

#
# STEP 1: SET the GRID!
#

Nt = 100 # number of time steps
Nx1, Nx2 = 75, 75  # nodes
sigma = 0.02
x1min, x1max = 0, 1 # physic domain x
x2min, x2max = 0, 1 # physic domain y

Lx1, Lx2 = int(abs(x1max - x1min)), int(abs(x2max - x2min)) #logic domain lenght 
dx1 = Lx1/(Nx1 - 1) # faces are (nodes - 1)
dx2 = Lx2/(Nx2 - 1)
#dt = sigma*dx # CFL condition
dt = 0.00005 

print("LATTICE PARAMETERS!")
print("Numbers of Iterations:", Nt)
print("Domain Length (Lx1, Lx2):", Lx1, Lx2)
print("Number of Nodes (Nx1, Nx2):", Nx1, Nx2)
print("Time Step:", dt)
print("Increments (dx1, dx2):", dx1, dx2)

#
# STEP 2: DEF. MESH GRID & PHYSICAL PARAM.!
#

x1 = np.linspace(x1min, x1max, Nx1, dtype=float)
x2 = np.linspace(x2min, x2max, Nx2, dtype=float)
x1v, x2v = np.meshgrid(x1, x2, indexing='ij')

# Initialization of field matrices
Ex1 = np.zeros([Nx1, Nx2], dtype=float)
Ex2 = np.zeros([Nx1, Nx2], dtype=float)
Bx1 = np.zeros([Nx1, Nx2], dtype=float)
Bx2 = np.zeros([Nx1, Nx2], dtype=float)
Bx3 = np.zeros([Nx1, Nx2], dtype=float)
Bx3old = np.zeros([Nx1, Nx2], dtype=float)
# Initialization of the Metric Tensor and the Jacobian
J = np.ones([Nx1, Nx2], dtype=float) 
gx1x1 = np.ones([Nx1, Nx2], dtype=float)
gx1x2 = np.zeros([Nx1, Nx2], dtype=float)
gx1x3 = np.zeros([Nx1, Nx2], dtype=float)
gx2x1 = np.zeros([Nx1, Nx2], dtype=float)
gx2x2 = np.ones([Nx1, Nx2], dtype=float)
gx2x3 = np.zeros([Nx1, Nx2], dtype=float)
gx3x1 = np.zeros([Nx1, Nx2], dtype=float)
gx3x2 = np.zeros([Nx1, Nx2], dtype=float)
gx3x3 = np.ones([Nx1, Nx2], dtype=float)

U = np.zeros([Nt], dtype=float) # Total energy
divE = np.zeros([Nt], dtype=float) # Divergence of E
Bx3[int((Nx1-1)/2),int((Nx2-1)/2)] = 0.001 # Initial condition

def periodicBC(A):
    A_w = np.zeros([Nx1, Nx2], dtype=float)
    A_s = np.zeros([Nx1, Nx2], dtype=float)
    A_b = np.zeros([Nx1, Nx2], dtype=float)
    # swop var.
    A_w[0, :] = A[0, :]
    A_s[:, 0] = A[:, 0]
    # Reflective BC for A field
    A[0, :]  = A[-1, :]   # west = est
    A[-1, :] = A_w[0, :]  # est = west
    A[:, 0]  = A[:, -1]   # south = north
    A[:, -1] = A_s[:, 0]  # north = south

def fields(Ex1, Ex2, Bx3):
    Bx3old[:, :] = Bx3[:, :]
    Ex1[ib:ie, jb:je] += dt * (1./(dx2*J[ib:ie, jb:je]))\
                                * (gx3x3[ib:ie, jb+1:je+1]*Bx3[ib:ie, jb+1:je+1] - gx3x3[ib:ie, jb:je]*Bx3[ib:ie, jb:je])
    Ex2[ib:ie, jb:je] -= dt * (1./(dx1*J[ib:ie, jb:je]))\
                                * (gx3x3[ib+1:ie+1, jb:je]*Bx3[ib+1:ie+1, jb:je] - gx3x3[ib:ie, jb:je]*Bx3[ib:ie, jb:je])
    periodicBC(Ex1)
    periodicBC(Ex2) 
    Bx3[ib:ie, jb:je] -= dt * ((1./(2.*dx1*J[ib:ie, jb:je]))\
                                *    (gx2x1[ib+1:ie+1, jb:je]*Ex1[ib+1:ie+1, jb:je] - gx2x1[ib-1:ie-1, jb:je]*Ex1[ib-1:ie-1, jb:je]\
                                + 2.*(gx2x2[ib:ie, jb:je]*Ex2[ib:ie, jb:je] - gx2x2[ib-1:ie-1, jb:je]*Ex2[ib-1:ie-1, jb:je]))\
                                - (1./(2.*dx2*J[ib:ie, jb:je]))\
                                * (2.*(gx1x1[ib:ie, jb:je]*Ex1[ib:ie, jb:je] - gx1x1[ib:ie, jb-1:je-1]*Ex1[jb:je, ib-1:ie-1])\
                                +      gx1x2[ib:ie, jb+1:je+1]*Ex2[ib:ie, jb+1:je+1] - gx1x2[ib:ie, jb-1:je-1]*Ex2[ib:ie, jb-1:je-1]))
    periodicBC(Bx3)
    return Ex1, Ex2, Bx3

#
# STEP 3: TIME EVOLUTION OF THE FIELD ON THE GRID!
#

# Start & End
# Note we don't start from zero cause 0 and Nx1-1 are the same node
ib = 1
jb = 1
ie = Nx1 - 1
je = Nx2 - 1

print("START SYSTEM EVOLUTION!")
for t in range(Nt): # count {0, Nt-1}
    print("Time steps:", t)
    Bx3old[:, :] = Bx3[:, :]
    '''
    # BEGIN : spatial update loops for Ey and Ex fields
    Ex1[ib:ie, jb:je] += dt * (1./(dx2*J[ib:ie, jb:je]))\
                                * (gx3x3[ib:ie, jb+1:je+1]*Bx3[ib:ie, jb+1:je+1] - gx3x3[ib:ie, jb:je]*Bx3[ib:ie, jb:je])
    Ex2[ib:ie, jb:je] -= dt * (1./(dx1*J[ib:ie, jb:je]))\
                                * (gx3x3[ib+1:ie+1, jb:je]*Bx3[ib+1:ie+1, jb:je] - gx3x3[ib:ie, jb:je]*Bx3[ib:ie, jb:je])
    # END : spatial update loops for Ex1 and Ex2 fields
    
    # swop var.
    Ex1_l[0, :] = Ex1[0, :]
    Ex1_b[:, 0] = Ex1[:, 0]
    Ex2_l[0, :] = Ex2[0, :]
    Ex2_b[:, 0] = Ex2[:, 0]
    # Reflective BC for E field
    Ex1[0, :] = Ex1[-1, :]   # left = right
    Ex1[-1, :] = Ex1_l[0, :] # right = left
    Ex1[:, 0] = Ex1[:, -1]   # bottom = top
    Ex1[:, -1] = Ex1_b[:, 0] # top = bottom
    Ex2[0, :] = Ex2[-1, :]   # left = right
    Ex2[-1, :] = Ex2_l[0, :] # right = left
    Ex2[:, 0] = Ex2[:, -1]   # bottom = top
    Ex2[:, -1] = Ex2_b[:,0]  # top = bottom
    
    # BEGIN : spatial update loops for Bz fields
    Bx3[ib:ie, jb:je] -= dt * ((1./(2.*dx1*J[ib:ie, jb:je]))\
                                *    (gx2x1[ib+1:ie+1, jb:je]*Ex1[ib+1:ie+1, jb:je] - gx2x1[ib-1:ie-1, jb:je]*Ex1[ib-1:ie-1, jb:je]\
                                + 2.*(gx2x2[ib:ie, jb:je]*Ex2[ib:ie, jb:je] - gx2x2[ib-1:ie-1, jb:je]*Ex2[ib-1:ie-1, jb:je]))\
                                - (1./(2.*dx2*J[ib:ie, jb:je]))\
                                * (2.*(gx1x1[ib:ie, jb:je]*Ex1[ib:ie, jb:je] - gx1x1[ib:ie, jb-1:je-1]*Ex1[jb:je, ib-1:ie-1])\
                                +      gx1x2[ib:ie, jb+1:je+1]*Ex2[ib:ie, jb+1:je+1] - gx1x2[ib:ie, jb-1:je-1]*Ex2[ib:ie, jb-1:je-1]))
    # END : spatial update loops for Bz fields
    
    # swop var.
    Bx3_l[0, :] = Bx3[0, :]
    Bx3_b[:, 0] = Bx3[:, 0]
    # Reflective BC for B field
    Bx3[0, :] = Bx3[-1, :]   # left = right
    Bx3[-1, :] = Bx3_l[0, :] # right = left
    Bx3[:, 0] = Bx3[:, -1]   # bottom = top
    Bx3[:, -1] = Bx3_b[:, 0] # top = bottom
    '''
    fields(Ex1, Ex2, Bx3)

    # Poynting theorem
    U[t] =  0.5 * np.sum(np.power(Ex1[ib:ie, jb:je],2.)\
                      + np.power(Ex2[ib:ie, jb:je],2.)\
                      + Bx3[ib:ie, jb:je] * Bx3old[ib:ie, jb:je])
    divE[t] = np.sum(1./J[ib:ie, jb:je]*((1/dx1) * (J[ib+1:ie+1, jb:je]*Ex1[ib+1:ie+1, jb:je] - J[ib:ie, jb:je]*Ex1[ib:ie, jb:je])\
                                           + (1/dx2) * (J[ib:ie, jb+1:je+1]*Ex2[ib:ie, jb+1:je+1] - J[ib:ie, jb:je]*Ex2[ib:ie, jb:je])))   
print("DONE!")

#
# STEP 4: VISUALIZATION!
#

#Plot3D(x1v, x2v, Bx3)
Plot2D(x1v, x2v, Bx3)
Energy(U)
DivE(divE)

plt.show()
