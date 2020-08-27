#
# created:        01.08.2020
# last modified:  25.08.2020
# author:         Luca Pezzini
# e-mail :        luca.pezzini@edu.unito.it
# MIT license

#
# Please feel free to use and modify this, but keep the above information. 
# Thanks!
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
from matplotlib import animation as anim
from mpl_toolkits.mplot3d import Axes3D

# Function def.

def energy(p):
    plt.figure()
    plt.plot(p)
    plt.title("Total Energy in Time")
    plt.xlabel("time")
    plt.ylabel("Utot")

def plot2D(x, y, p):
    plt.figure()
    plt.contourf(x, y, p)#, cmap=plt.cm.jet)
    plt.title("Contour of Bz")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.colorbar()

def plot3D(x, y, p):
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, p[:], rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False)
    ax.view_init(30, 225)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.show()
    
# Animation setup
#fig = plt.figure('frame')
#plt.rcParams['animation.ffmpeg_path'] = "/Users/luca_pezzini/Documents/00-plasma_phy/covariant_pic-2d"
#plt.rcParams['animation.bitrate'] = 100000
#ims = []

#
# STEP 1: SET the GRID!
#

Nt = 4000 # number of time steps
Nx, Ny = 501, 501  # nodes
sigma = 0.2
xmin, xmax = 0, 1 # physic domain x
ymin, ymax = 0, 1 # physic domain y

Lx, Ly = int(abs(xmax-xmin)), int(abs(ymax-ymin)) #logic domain lenght 
dx = Lx/(Nx-1) # faces are (nodes - 1)
dy = Ly/(Ny-1)
dt = sigma*dx # CFL condition

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

x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
xv, yv = np.meshgrid(x, y, indexing='ij')

# Initialization of field matrices
Bz = np.zeros([Nx, Ny])
Ex = np.zeros([Nx, Ny])
Ey = np.zeros([Nx, Ny])
Bzbef = np.zeros([Nx, Ny])
Utot = np.zeros([Nt])

# Swop variable
Ex_l = np.zeros([Nx, Ny])
Ex_b = np.zeros([Nx, Ny])
Ey_l = np.zeros([Nx, Ny])
Ey_b = np.zeros([Nx, Ny])
Bz_l = np.zeros([Nx, Ny])
Bz_b = np.zeros([Nx, Ny])

# Initial conditions
Bz[int((Nx-1)/2),int((Ny-1)/2)] = 0.001

#
# STEP 3: TIME EVOLUTION OF THE FIELD ON THE GRID!
#

# Start & End
# Note we don't start from zero cause 0 and Nx-1 are the same node
xs = 1
ys = 1
xe = Nx-1
ye = Ny-1

print("START SYSTEM EVOLUTION!")
for t in range(Nt): # count {0, Nt-1}

    # BEGIN : spatial update loops for Ey and Ex fields
    Ex[xs:xe, ys:ye] +=   (dt/dy) * (Bz[xs:xe, ys+1:ye+1] - Bz[xs:xe, ys:ye])
    Ey[xs:xe, ys:ye] += - (dt/dx) * (Bz[xs+1:xe+1, ys:ye] - Bz[xs:xe, ys:ye])
    #Ex[xs+1:xe+1, ys:ye] +=   (dt/dy) * (Bz[xs+1:xe+1, ys+1:ye+1] - Bz[xs+1:xe+1, ys:ye])
    #Ey[xs:xe, ys+1:ye+1] += - (dt/dx) * (Bz[xs+1:xe+1, ys+1:ye+1] - Bz[xs:xe, ys+1:ye+1])
    # END : spatial update loops for Ey and Ex fields
    
    # swop var.
    Ex_l[0, :] = Ex[0, :]
    Ex_b[:, 0] = Ex[:, 0]
    Ey_l[0, :] = Ey[0, :]
    Ey_b[:, 0] = Ey[:, 0]
    # Reflective BC for E field
    Ex[0, :]   = Ex[-1, :]  #left=right
    Ex[-1, :]  = Ex_l[0, :] #right=left
    Ex[:, 0]   = Ex[:, -1]  #bottom=top
    Ex[:, -1]  = Ex_b[:, 0] #top=bottom
    Ey[0, :]   = Ey[-1, :] #left=right
    Ey[-1, :]  = Ey_l[0, :] #right=left
    Ey[:, 0]   = Ey[:, -1] #bottom=top
    Ey[:, -1]  = Ey_b[:,0] #top=bottom

    Bzbef[:, :] = Bz[:, :]

    # BEGIN : spatial update loops for Bz fields
    # NOTE: B field is calculate with the updated E field so is half a step ahead
    Bz[xs:xe, ys:ye] += dt * ((1/dy) * (Ex[xs:xe, ys:ye] - Ex[xs:xe, ys-1:ye-1])\
                            - (1/dx) * (Ey[xs:xe, ys:ye] - Ey[xs-1:xe-1, ys:ye]))
    #Bz[xs+1:xe+1, ys+1:ye+1] += dt * ((1/dy) * (Ex[xs+1:xe+1, ys+1:ye+1] - Ex[xs+1:xe+1, ys:ye])\
    #                       - (1/dx) * (Ey[xs+1:xe+1, ys+1:ye+1] - Ey[xs:xe, ys+1:ye+1]))
    # END : spatial update loops for Bz fields
    
    # Reflective BC for B field
    Bz_l[0, :] = Bz[0, :]
    Bz_b[:, 0] = Bz[:, 0]

    Bz[0, :]   = Bz[-1, :] #left=right
    Bz[-1, :]  = Bz_l[0, :] #right=left
    Bz[:, 0]   = Bz[:, -1] #bottom=top
    Bz[:, -1]  = Bz_b[:, 0] #top=bottom

    # Energy must conserve
    # Time is staggered to updating E and B. So to calculate Utot 
    # we should center the value of the B field to the same point in 
    # time (i.e. B_av(t) = (B(t) + B(t+1))/2)

    Utot[t] = np.sum(np.power(Ex[xs:xe, ys:ye],2.)\
                      + np.power(Ey[xs:xe, ys:ye],2.)\
                      + np.power((Bz[xs:xe, ys:ye] + Bzbef[xs:xe, ys:ye])/2.,2.))
print("DONE!")

#
# STEP 4: VISUALIZATION!
#

# Create figures
#plot3D(xv, yv, Bz)
plot2D(xv, yv, Bz)
energy(Utot)

# Create animation
#an = anim.ArtistAnimation(fig, ims, interval=10, repeat_delay=0, blit=True)
#writer = anim.FFMpegWriter(fps=30)
#an.save('bz_vid.mp4', writer=writer, dpi=500)

plt.show()
