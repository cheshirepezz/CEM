#
# created:        01.08.2020
# last modified:  08.09.2020
# author:         Luca Pezzini
# e-mail :        luca.pezzini@edu.unito.it
# MIT license

#
# Please feel free to use and modify this code, but keep the above information. 
# Thanks!
#

#
#                2D CARTESIAN TE_z MODE ON x2EE MESH (FDTD)
# Solve the Transverse Electric mode equation on the xy-plane (TE_z)
# using the Finite Difference Time Domain metod (FDTD) on the x2ee lattice
# in simple cartesian geometry. 
#           x2ee Grid (Staggered grid)
#           E[n-1] H[n-1/2] E[n] H[n+1/2] E[n+1]
#           E[0          1          2          3]
#           B[      0          1          2     ]
# The E fields are defined at every unitary space indices (i, j); instead
# B field is defined at fractional indices (i+0.5, j+0.5) as the x2ee
# It's used the convention to shift i,j+1/2->i,j+1 & n+1/2->n solving the 
# problem of representing fractional indices.
# Phx2sical parameter are converted in undimensional code units.
# The time update is done using Leapfrog time-stepping. Here, E-fields
# i.e Ex and Ex2 are updated every full time-step and Bx3 field is updated every 
# half time-step. This is shown by three alternating for-loop updates in groups 
# of two (for E) and one (for B) spanning entire spatial grid inside a main 
# for-loop for time update spanning the entire time-grid.
#

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Ax1es3D
from matplotlib import animation as anim
import time
import gc

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
    plt.title("Contour of Bx3")
    plt.xlabel("y")
    plt.ylabel("x")
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

animate = True
save = False
    
# Bx3 animation setup
if animate:    
    fig = plt.figure('Bx3 animation')    
    plt.rcParams['animation.ffmpeg_path'] = "/usr/local/Cellar/ffmpeg/4.2.2_2/bin"    
    plt.rcParams['animation.bitrate'] = 100000    
    ims = []

#
# STEP 1: SET the GRID!
#

Nt = 4000 # number of time steps
Nx1, Nx2 = 50, 50  # nodes
sigma = 0.02
x1min, x1max = 0, 1 # phx2sic domain x1
x2min, x2max = 0, 1 # phx2sic domain x2

Lx1, Lx2 = int(abs(x1max - x1min)), int(abs(x2max - x2min)) #logic domain lenght 
dx1 = Lx1/(Nx1 - 1) # faces are (nodes - 1)
dx2 = Lx2/(Nx2 - 1)
#dt = sigma*dx1 # CFL condition
dt = 0.00005 

print("LATTICE PARAMETERS!")
print("Numbers of Iterations:", Nt)
print("Domain Length (Lx1, Lx2):", Lx1, Lx2)
print("Number of Nodes (Nx1, Nx2):", Nx1, Nx2)
print("Time Step:", dt)
print("Increments (dx11, dx12):", dx1, dx2)

#
# STEP 2: DEF. MESH GRID & PHx2SICAL PARAM.!
#

# Mesh grid
x1 = np.linspace(x1min, x1max, Nx1, dtype=float)
x2 = np.linspace(x2min, x2max, Nx2, dtype=float)
x1v, x2v = np.meshgrid(x1, x2, indexing='ij')

# Geometry
h1 = 1.0
h2 = x1v
h3 = 1.0

# Initialization of field matrices
Ex1 = np.zeros([Nx1, Nx2], dtype=float)
Ex2 = np.zeros([Nx1, Nx2], dtype=float)
Bx3 = np.zeros([Nx1, Nx2], dtype=float)
Bx3old = np.zeros([Nx1, Nx2], dtype=float)

# Swop variable
Ex1_l = np.zeros([Nx1, Nx2], dtype=float)
Ex1_b = np.zeros([Nx1, Nx2], dtype=float)
Ex2_l = np.zeros([Nx1, Nx2], dtype=float)
Ex2_b = np.zeros([Nx1, Nx2], dtype=float)
Bx3_l = np.zeros([Nx1, Nx2], dtype=float)
Bx3_b = np.zeros([Nx1, Nx2], dtype=float)

U = np.zeros([Nt], dtype=float) # Total energy
divE = np.zeros([Nt], dtype=float) # Divergence of E
Bx3[int((Nx1 - 1)/2.),int((Nx2 - 1)/2.)] = 0.001 # Initial conditions

#
# STEP 3: TIME EVOLUTION OF THE FIELD ON THE GRID!
#

# Start & End
# Note we don't start from zero cause 0 and Nx1-1 are the same node
x1s = 1
x2s = 1
x1e = Nx1 - 1
x2e = Nx2 - 1

print("START SYSTEM EVOLUTION!")
start = time.time()
for t in range(Nt): # count {0, Nt-1}

    Bx3old[:, :] = Bx3[:, :]

    # BEGIN : spatial update loops for Ex2 and Ex fields
    Ex1[x1s:x1e, x2s:x2e] += dt * ((1./(dx2*h2)) * (Bx3[x1s:x1e, x2s+1:x2e+1] - Bx3[x1s:x1e, x2s:x2e]))
    Ex2[x1s:x1e, x2s:x2e] -= dt * ((1./(dx1*h1)) * (Bx3[x1s+1:x1e+1, x2s:x2e] - Bx3[x1s:x1e, x2s:x2e]))   
    # END : spatial update loops for Ex2 and Ex fields
    
    # swop var.
    Ex1_l[0, :] = Ex1[0, :]
    Ex1_b[:, 0] = Ex1[:, 0]
    Ex2_l[0, :] = Ex2[0, :]
    Ex2_b[:, 0] = Ex2[:, 0]
    # Reflective BC for E field
    Ex1[0, :]  = Ex1[-1, :]  # left = right
    Ex1[-1, :] = Ex1_l[0, :] # right = left
    Ex1[:, 0]  = Ex1[:, -1]  # bottom = top
    Ex1[:, -1] = Ex1_b[:, 0] # top = bottom
    Ex2[0, :]  = Ex2[-1, :]  # left = right
    Ex2[-1, :] = Ex2_l[0, :] # right = left
    Ex2[:, 0]  = Ex2[:, -1]  # bottom = top
    Ex2[:, -1] = Ex2_b[:, 0]  # top = bottom

    # BEGIN : spatial update loops for Bx3 fields
    # note: B field is calculate with the updated E field so is half a step ahead
    Bx3[x1s:x1e, x2s:x2e] += dt * ((1./(dx2*h2) * (Ex1[x1s:x1e, x2s:x2e] - Ex1[x1s:x1e, x2s-1:x2e-1])\
                                 - (1./(dx1*h1) * (Ex2[x1s:x1e, x2s:x2e] - Ex2[x1s-1:x1e-1, x2s:x2e]))
    # END : spatial update loops for Bx3 fields

    # swop var.
    Bx3_l[0, :] = Bx3[0, :]
    #Bx3_b[:, 0] = Bx3[:, 0]
    # Reflective BC for B field
    #Bx3[0, :]   = Bx3[-1, :]  # left = right
    #Bx3[-1, :]  = Bx3_l[0, :] # right = left
    #Bx3[:, 0]   = Bx3[:, -1]  # bottom = top
    #Bx3[:, -1]  = Bx3_b[:, 0] # top = bottom
    

    # Poynting theorem
    #U[t] =  0.5 * np.sum(np.power(Ex[x1s:x1e, x2s:x2e],2.)\
                      + np.power(Ex2[x1s:x1e, x2s:x2e],2.)\
                      + Bx3[x1s:x1e, x2s:x2e] * Bx3old[x1s:x1e, x2s:x2e])
    #U[t] =  0.5 * np.sum(Bx3[x1s:x1e, x2s:x2e] * Bx3old[x1s:x1e, x2s:x2e])
    #divE[t] = np.sum((1./dx1) * (Ex1[x1s+1:x1e+1, x2s:x2e] - Ex1[x1s:x1e, x2s:x2e])\
                   + (1./dx2) * (Ex2[x1s:x1e, x2s+1:x2e+1] - Ex2[x1s:x1e, x2s:x2e]))

    # Animation frame gathering
    if (t % 20 == 0) & animate:
        print(t/Nt)
        im = plt.imshow(Bx3, origin='lower', extent=[0, Lx1, 0, Lx2], aspect='equal', vmin=-0.1, vmax=0.1)
        ims.append([im])
    gc.collect()
    
stop = time.time()
print("DONE!")

print("Min Energy:", np.min(U))
print("Max Energy:", np.max(U))

#
# STEP 4: VISUALIZATION!
#

if animate:
    an = anim.ArtistAnimation(fig, ims, interval=1, repeat_delay=0, blit=True)
    writer = anim.FFMpegWriter(fps=30)
    if save:
        an.save('bx3_vid.mp4', writer=writer, dpi=500)

#Plot3D(x1v, x2v, Bx3)
Plot2D(x1v, x2v, Bx3)
Energy(U)
DivE(divE)

plt.show()
