#
# created:        08.09.2020
# last modified:  09.09.2020
# author:         Luca Pezzini
# e-mail :        luca.pezzini@edu.unito.it
# MIT license

#
# Please feel free to use and modify this, but keep the above information. 
# Thanks!
#

#
#             3D CARTESIAN MAXWELL EQUATION ON YEE MESH (FDTD)
# Solve the Maxwell equation using the Finite Difference 
# Time Domain metod (FDTD) on the Yee lattice in simple cartesian geometry. 
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
import sys
import numpy as np
import matplotlib.pyplot as plt

def Energy(p):
    plt.figure()
    plt.plot(p)
    plt.title("Energy Conservation")
    plt.xlabel("time")
    plt.ylabel("U")

def DivB(p):
    plt.figure()
    plt.plot(p)
    plt.title("Mimetic Operator")
    plt.xlabel("time")
    plt.ylabel("div(B)")

#
# STEP 1: SET the GRID!
#

Nt = 1000 # number of time steps
Nx, Ny, Nz = 50, 50, 50  # nodes
sigma = 0.02
xmin, xmax = 0, 1 # physic domain x
ymin, ymax = 0, 1 # physic domain y
zmin, zmax = 0, 1 # physic domain z

Lx = int(abs(xmax - xmin)) # logic domain lenght x
Ly = int(abs(ymax - ymin)) # logic domain lenght y
Lz = int(abs(zmax - zmin)) # logic domain lenght z
dx = Lx/(Nx - 1) # faces are (nodes - 1)
dy = Ly/(Ny - 1)
dz = Ly/(Nz - 1)
dt = sigma*dx # CFL condition
#dt = 0.00005 

print("LATTICE PARAMETERS!")
print("Numbers of Iterations:", Nt)
print("Domain Length (Lx, Ly, Lz):", Lx, Ly, Lz)
print("Number of Nodes (Nx, Ny, Nz):", Nx, Ny, Nz)
print("Time Step:", dt)
print("Increments (dx, dy, dz):", dx, dy, dz)

#
# STEP 2: DEF. MESH GRID & PHYSICAL PARAM.!
#

x = np.linspace(xmin, xmax, Nx, dtype=float)
y = np.linspace(ymin, ymax, Ny, dtype=float)
z = np.linspace(zmin, zmax, Nz, dtype=float)
xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
xv1, yv1 = np.meshgrid(x, y, indexing='ij')

# Initialization of field matrices
Ex = np.zeros([Nx, Ny, Nz], dtype=float)
Ey = np.zeros([Nx, Ny, Nz], dtype=float)
Ez = np.zeros([Nx, Ny, Nz], dtype=float)
Bx = np.zeros([Nx, Ny, Nz], dtype=float)
By = np.zeros([Nx, Ny, Nz], dtype=float)
Bz = np.zeros([Nx, Ny, Nz], dtype=float)
Bxold = np.zeros([Nx, Ny, Nz], dtype=float)
Byold = np.zeros([Nx, Ny, Nz], dtype=float)
Bzold = np.zeros([Nx, Ny, Nz], dtype=float)
Exy = np.zeros([Nx, Nz], dtype=float)
Eyz = np.zeros([Nx, Nz], dtype=float)
Exz = np.zeros([Nx, Nz], dtype=float)
Bxy = np.zeros([Nx, Nz], dtype=float)
Byz = np.zeros([Nx, Nz], dtype=float)
Bxz = np.zeros([Nx, Nz], dtype=float)
Emax = []
Bmax = []

# Swop variable
Ex_w = np.zeros([Nx, Ny, Nz], dtype=float)
Ex_s = np.zeros([Nx, Ny, Nz], dtype=float)
Ex_b = np.zeros([Nx, Ny, Nz], dtype=float)
Ey_w = np.zeros([Nx, Ny, Nz], dtype=float)
Ey_s = np.zeros([Nx, Ny, Nz], dtype=float)
Ey_b = np.zeros([Nx, Ny, Nz], dtype=float)
Ez_w = np.zeros([Nx, Ny, Nz], dtype=float)
Ez_s = np.zeros([Nx, Ny, Nz], dtype=float)
Ez_b = np.zeros([Nx, Ny, Nz], dtype=float)
Bx_w = np.zeros([Nx, Ny, Nz], dtype=float)
Bx_s = np.zeros([Nx, Ny, Nz], dtype=float)
Bx_b = np.zeros([Nx, Ny, Nz], dtype=float)
By_w = np.zeros([Nx, Ny, Nz], dtype=float)
By_s = np.zeros([Nx, Ny, Nz], dtype=float)
By_b = np.zeros([Nx, Ny, Nz], dtype=float)
Bz_w = np.zeros([Nx, Ny, Nz], dtype=float)
Bz_s = np.zeros([Nx, Ny, Nz], dtype=float)
Bz_b = np.zeros([Nx, Ny, Nz], dtype=float)

Bz[int((Nx - 1)/2),int((Ny - 1)/2), int((Nz - 1)/2)] = 1. # Initial conditions
U = np.zeros([Nt], dtype=float) # Total energy
divB = np.zeros([Nt], dtype=float) # Divergence of B

#
# STEP 3: TIME EVOLUTION OF THE FIELD ON THE GRID!
#

#flag
flag_plt = False
flag_start = False

# Start & End
# Note we don't start from zero cause 0 and Nx-1 are the same node
xs, xe = 1, Nx - 1
ys, ye = 1, Ny - 1
zs, ze = 1, Nz - 1

print("START SYSTEM EVOLUTION!")
for t in range(Nt): # count {0, Nt-1}

    Bxold[:, :, :] = Bx[:, :, :]
    Byold[:, :, :] = By[:, :, :]
    Bzold[:, :, :] = Bz[:, :, :]

    # BEGIN : spatial update loops for Ey and Ex fields
    Ex[xs:xe, ys:ye, zs:ze] += (dt/dy) * (Bz[xs:xe, ys+1:ye+1, zs:ze] - Bz[xs:xe, ys:ye, zs:ze])\
                             - (dt/dz) * (By[xs:xe, ys:ye, zs+1:ze+1] - By[xs:xe, ys:ye, zs:ze])
    Ey[xs:xe, ys:ye, zs:ze] += (dt/dz) * (Bx[xs:xe, ys:ye, zs+1:ze+1] - Bx[xs:xe, ys:ye, zs:ze])\
                             - (dt/dx) * (Bz[xs+1:xe+1, ys:ye, zs:ze] - Bz[xs:xe, ys:ye, zs:ze])
    Ez[xs:xe, ys:ye, zs:ze] += (dt/dx) * (By[xs+1:xe+1, ys:ye, zs:ze] - By[xs:xe, ys:ye, zs:ze])\
                             - (dt/dy) * (Bx[xs:xe, ys+1:ye+1, zs:ze] - Bx[xs:xe, ys:ye, zs:ze])

    # swop var.
    Ex_w[0, :, :] = Ex[0, :, :]
    Ex_s[:, 0, :] = Ex[:, 0, :]
    Ex_b[:, :, 0] = Ex[:, :, 0]
    Ey_w[0, :, :] = Ey[0, :, :]
    Ey_s[:, 0, :] = Ey[:, 0, :]
    Ey_b[:, :, 0] = Ey[:, :, 0]
    Ez_w[0, :, :] = Ez[0, :, :]
    Ez_s[:, 0, :] = Ez[:, 0, :]
    Ez_b[:, :, 0] = Ez[:, :, 0]
    # Reflective BC for E field
    Ex[0, :, :] = Ex[-1, :, :]   # west = est
    Ex[-1, :, :] = Ex_w[0, :, :] # est = west
    Ex[:, 0, :] = Ex[:, -1, :]   # south = north
    Ex[:, -1, :] = Ex_s[:, 0, :] # north = south
    Ex[:, :, 0] = Ex[:, :, -1]   # bottom = top
    Ex[:, :, -1] = Ex_b[:, :, 0] # top = bottom   
    Ey[0, :, :] = Ey[-1, :, :]   # west = est
    Ey[-1, :, :] = Ey_w[0, :, :] # est = west
    Ey[:, 0, :] = Ey[:, -1, :]   # south = north
    Ey[:, -1, :] = Ey_s[:, 0, :] # north = south
    Ey[:, :, 0] = Ey[:, :, -1]   # bottom = top
    Ey[:, :, -1] = Ey_b[:, :, 0] # top = bottom
    Ez[0, :, :] = Ey[-1, :, :]   # west = est
    Ez[-1, :, :] = Ey_w[0, :, :] # est = west
    Ez[:, 0, :] = Ey[:, -1, :]   # south = north
    Ez[:, -1, :] = Ey_s[:, 0, :] # north = south
    Ez[:, :, 0] = Ey[:, :, -1]   # bottom = top
    Ez[:, :, -1] = Ey_b[:, :, 0] # top = bottom
 
    # BEGIN : spatial update loops for Bz fields
    # NOTE: B field is calculate with the updated E field so is half a step ahead
    Bx[xs:xe, ys:ye, zs:ze] += dt * ((1/dz) * (Ey[xs:xe, ys:ye, zs:ze] - Ey[xs:xe, ys:ye, zs-1:ze-1])\
                                   - (1/dy) * (Ez[xs:xe, ys:ye, zs:ze] - Ez[xs:xe, ys-1:ye-1, zs:ze]))
    By[xs:xe, ys:ye, zs:ze] += dt * ((1/dx) * (Ez[xs:xe, ys:ye, zs:ze] - Ez[xs-1:xe-1, ys:ye, zs:ze])\
                                   - (1/dz) * (Ex[xs:xe, ys:ye, zs:ze] - Ex[xs:xe, ys:ye, zs-1:ze-1]))
    Bz[xs:xe, ys:ye, zs:ze] += dt * ((1/dy) * (Ex[xs:xe, ys:ye, zs:ze] - Ex[xs:xe, ys-1:ye-1, zs:ze])\
                                   - (1/dx) * (Ey[xs:xe, ys:ye, zs:ze] - Ey[xs-1:xe-1, ys:ye, zs:ze])) 
    # END : spatial update loops for Bz fields
 
    # swop var.
    Bz_w[0, :, :] = Bz[0, :, :] # west
    Bz_s[:, 0, :] = Bz[:, 0, :] # south
    Bz_b[:, :, 0] = Bz[:, :, 0] # bottom
    # Reflective BC for B field
    Bz[0, :, :]   = Bz[-1, :, :] # west = est
    Bz[-1, :, :]  = Bz_w[0, :, :] # est = west
    Bz[:, 0, :]   = Bz[:, -1, :] # south = north
    Bz[:, -1, :]  = Bz_s[:, 0, :] # north = south
    Bz[:, :, 0]   = Bz[:, :, -1] # bottom = top
    Bz[:, :, -1]  = Bz_b[:, :, 0] # top = bottom

    # Energy must conserve
    # Time is staggered to updating E and B. So to calculate U 
    # we should center the value of the B field to the same point in 
    # time (i.e. B_av(t) = (B(t) + B(t+1))/2)

    U[t] =  0.5 * np.sum(np.power(Ex[xs:xe, ys:ye, zs:ze],2.)\
                          + np.power(Ey[xs:xe, ys:ye, zs:ze],2.)\
                          + np.power(Ez[xs:xe, ys:ye, zs:ze],2.)\
                          + Bx[xs:xe, ys:ye, zs:ze] * Bxold[xs:xe, ys:ye, zs:ze]\
                          + By[xs:xe, ys:ye, zs:ze] * Byold[xs:xe, ys:ye, zs:ze]\
                          + Bz[xs:xe, ys:ye, zs:ze] * Bzold[xs:xe, ys:ye, zs:ze])
    divB[t] = np.sum((1/dx) * (Bx[xs+1:xe+1, ys:ye, zs:ze] - Bx[xs:xe, ys:ye, zs:ze])\
                   + (1/dy) * (By[xs:xe, ys+1:ye+1, zs:ze] - By[xs:xe, ys:ye, zs:ze])\
                   + (1/dz) * (Bz[xs:xe, ys:ye, zs+1:ze+1] - Bz[xs:xe, ys:ye, zs:ze]))
    # Reduce the dimension slicing in the middle of the plane y es.
    Exy[xs:xe, zs:ze] = np.power(np.power(Ex[xs:xe, ys:ye, int(Ny/2.)], 2.)\
                               + np.power(Ey[xs:xe, ys:ye, int(Ny/2.)], 2.)\
                               + np.power(Ez[xs:xe, ys:ye, int(Ny/2.)], 2.) , 1./2.)
    Eyz[xs:xe, zs:ze] = np.power(np.power(Ex[int(Ny/2.), ys:ye, zs:ze], 2.)\
                               + np.power(Ey[int(Ny/2.), ys:ye, zs:ze], 2.)\
                               + np.power(Ez[int(Ny/2.), ys:ye, zs:ze], 2.) , 1./2.)
    Exz[xs:xe, zs:ze] = np.power(np.power(Ex[xs:xe, int(Ny/2.), zs:ze], 2.)\
                               + np.power(Ey[xs:xe, int(Ny/2.), zs:ze], 2.)\
                               + np.power(Ez[xs:xe, int(Ny/2.), zs:ze], 2.) , 1./2.)
    Bxy[xs:xe, zs:ze] = np.power(np.power(Bx[xs:xe, ys:ye, int(Ny/2.)], 2.)\
                               + np.power(By[xs:xe, ys:ye, int(Ny/2.)], 2.)\
                               + np.power(Bz[xs:xe, ys:ye, int(Ny/2.)], 2.) , 1./2.)
    Byz[xs:xe, zs:ze] = np.power(np.power(Bx[int(Ny/2.), ys:ye, zs:ze], 2.)\
                               + np.power(By[int(Ny/2.), ys:ye, zs:ze], 2.)\
                               + np.power(Bz[int(Ny/2.), ys:ye, zs:ze], 2.) , 1./2.)
    Bxz[xs:xe, zs:ze] = np.power(np.power(Bx[xs:xe, int(Ny/2.), zs:ze], 2.)\
                               + np.power(By[xs:xe, int(Ny/2.), zs:ze], 2.)\
                               + np.power(Bz[xs:xe, int(Ny/2.), zs:ze], 2.) , 1./2.)
    Emax.append([max(Exy.max(), Eyz.max(), Exz.max())])
    Bmax.append([max(Bxy.max(), Byz.max(), Bxz.max())])

    if flag_plt == False:
        plt.figure(figsize =(12, 10))
    
    plt.subplot(2,2,1)
    plt.pcolor(xv1, yv1, Bxy)
    plt.title("Bxy plane")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.pcolor(xv1, yv1, Byz)
    plt.title("Byz plane")
    plt.xlabel("y")
    plt.ylabel("z")
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.pcolor(xv1, yv1, Bxz)
    plt.title("Bxz plane")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.plot(Emax)
    plt.plot(Bmax)
    plt.title('Emax (t)')
    plt.xlabel("time")
    plt.ylabel("Fields max")
    
    if flag_start == False:
        command = input("Press Enter, then start.")
        flag_start = True
        
    plt.pause(0.001)
    plt.clf()
    flag_plt=True

print("DONE!")

print("Min Energy", np.min(U))
print("Max Energy", np.max(U))

#
# STEP 4: VISUALIZATION!
#

Energy(U)
DivE(divB)

plt.show()
