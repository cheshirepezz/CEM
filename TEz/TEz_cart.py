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
#                2D CARTESIAN TE_z MODE ON YEE MESH (FDTD)
# Solve the Transverse Electric mode equation on the xy-plane (TE_z)
# using the Finite Difference Time Domain metod (FDTD) on the Yee lattice
# in simple cartesian geometry. 
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
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
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
    plt.title("Contour of Bz")
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
    
# Bz animation setup
if animate:    
    fig = plt.figure('Bz animation')    
    plt.rcParams['animation.ffmpeg_path'] = "/usr/local/Cellar/ffmpeg/4.2.2_2/bin"    
    plt.rcParams['animation.bitrate'] = 100000    
    ims = []

#
# STEP 1: SET the GRID!
#

Nt = 3000 # number of time steps
Nx, Ny = 50, 50  # nodes
sigma = 0.02
xmin, xmax = 0, 1 # physic domain x
ymin, ymax = 0, 1 # physic domain y

Lx, Ly = int(abs(xmax - xmin)), int(abs(ymax - ymin)) #logic domain lenght 
dx = Lx/(Nx - 1) # faces are (nodes - 1)
dy = Ly/(Ny - 1)
#dt = sigma*dx # CFL condition
dt = 0.00005 

print("LATTICE PARAMETERS!")
print("Numbers of Iterations:", Nt)
print("Domain Length (Lx, Ly):", Lx, Ly)
print("Number of Nodes (Nx, Ny):", Nx, Ny)
print("Time Step:", dt)
print("Increments (dx, dy):", dx, dy)

#
# STEP 2: DEF. MESH GRID & PHYSICAL PARAM.!
# giving the string ‘ij’ returns a meshgrid with matrix
# indexing, while ‘xy’ returns a meshgrid with Cartesian indexing
#

x = np.linspace(xmin, xmax, Nx, dtype=float)
y = np.linspace(ymin, ymax, Ny, dtype=float)
xv, yv = np.meshgrid(x, y, indexing='ij')

# Initialization of field matrices
Ex = np.zeros([Nx, Ny], dtype=float)
Ey = np.zeros([Nx, Ny], dtype=float)
Bz = np.zeros([Nx, Ny], dtype=float)
Bzold = np.zeros([Nx, Ny], dtype=float)

# Swop variable
Ex_l = np.zeros([Nx, Ny], dtype=float)
Ex_b = np.zeros([Nx, Ny], dtype=float)
Ey_l = np.zeros([Nx, Ny], dtype=float)
Ey_b = np.zeros([Nx, Ny], dtype=float)
Bz_l = np.zeros([Nx, Ny], dtype=float)
Bz_b = np.zeros([Nx, Ny], dtype=float)

U = np.zeros([Nt], dtype=float) # Total energy
divE = np.zeros([Nt], dtype=float) # Divergence of E
Bz[int((Nx-1)/2),int((Ny-1)/2)] = 0.001 # Initial conditions

#
# STEP 3: TIME EVOLUTION OF THE FIELD ON THE GRID!
#

# Start & End
# Note we don't start from zero cause 0 and Nx-1 are the same node
xs = 1
ys = 1
xe = Nx - 1
ye = Ny - 1

print("START SYSTEM EVOLUTION!")
start = time.time()
for t in range(Nt): # count {0, Nt-1}

    Bzold[:, :] = Bz[:, :]

    # BEGIN : spatial update loops for Ey and Ex fields
    Ex[xs:xe, ys:ye] += dt * ((1/dy) * (Bz[xs:xe, ys+1:ye+1] - Bz[xs:xe, ys:ye]))
    Ey[xs:xe, ys:ye] -= dt * ((1/dx) * (Bz[xs+1:xe+1, ys:ye] - Bz[xs:xe, ys:ye]))   
    # END : spatial update loops for Ey and Ex fields
    
    # swop var.
    Ex_l[0, :] = Ex[0, :]
    Ex_b[:, 0] = Ex[:, 0]
    Ey_l[0, :] = Ey[0, :]
    Ey_b[:, 0] = Ey[:, 0]
    # Reflective BC for E field
    Ex[0, :] = Ex[-1, :]  #  left = right
    Ex[-1, :] = Ex_l[0, :] # right = left
    Ex[:, 0] = Ex[:, -1]  # bottom = top
    Ex[:, -1] = Ex_b[:, 0] # top = bottom
    Ey[0, :] = Ey[-1, :] #l eft = right
    Ey[-1, :] = Ey_l[0, :] # right = left
    Ey[:, 0] = Ey[:, -1] # bottom = top
    Ey[:, -1] = Ey_b[:,0] # top = bottom

    # BEGIN : spatial update loops for Bz fields
    # note: B field is calculate with the updated E field so is half a step ahead
    Bz[xs:xe, ys:ye] += dt * ((1/dy) * (Ex[xs:xe, ys:ye] - Ex[xs:xe, ys-1:ye-1])\
                            - (1/dx) * (Ey[xs:xe, ys:ye] - Ey[xs-1:xe-1, ys:ye]))
    # END : spatial update loops for Bz fields
    
    # swop var.
    Bz_l[0, :] = Bz[0, :]
    Bz_b[:, 0] = Bz[:, 0]
    # Reflective BC for B field
    Bz[0, :]   = Bz[-1, :] # left = right
    Bz[-1, :]  = Bz_l[0, :] # right = left
    Bz[:, 0]   = Bz[:, -1] # bottom = top
    Bz[:, -1]  = Bz_b[:, 0] # top = bottom

    # Poynting theorem
    U[t] =  0.5 * np.sum(np.power(Ex[xs:xe, ys:ye],2.)\
                      + np.power(Ey[xs:xe, ys:ye],2.)\
                      + Bz[xs:xe, ys:ye] * Bzold[xs:xe, ys:ye])
    divE[t] = np.sum((1/dx) * (Ex[xs+1:xe+1, ys:ye] - Ex[xs:xe, ys:ye])\
                   + (1/dy) * (Ey[xs:xe, ys+1:ye+1] - Ey[xs:xe, ys:ye]))

    # Animation frame gathering
    if (t % 20 == 0) & animate:
        print(t/Nt)
        im = plt.imshow(Bz, origin='lower', extent=[0, Lx, 0, Ly], aspect='equal', vmin=-0.1, vmax=0.1)
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
        an.save('bz_vid.mp4', writer=writer, dpi=500)

#Plot3D(xv, yv, Bz)
Plot2D(xv, yv, Bz)
Energy(U)
DivE(divE)

plt.show()
