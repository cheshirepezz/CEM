#
# author:         Luca Pezzini
# e-mail :        luca.pezzini@edu.unito.it
# created:        28.10.2020
# last modified:  28.10.2020
# MIT license
#

#
# Transverse Electromagnetics mode 2D 
#          • General coordinates
#          • Energy conserving
#          • Mimetic operator
#

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import time

# Time steps
Nt = 500 
dt = 0.001
# Nodes
Nx = 100
Ny = 100
Nz = 1

#flag
flag_plt = False
flag_start = False

# Computational domain 
xmin, xmax = 0, 1    
ymin, ymax = 0, 1    
zmin, zmax = 0, .25  
Lx = int(abs(xmax - xmin))
Ly = int(abs(ymax - ymin)) 
Lz = abs(zmax - zmin)
dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz

# Grids matrix
x = np.linspace(0, Lx-dx, Nx, dtype=float)
y = np.linspace(0, Ly-dy, Ny, dtype=float)
z = np.linspace(0, Lz-dz, Nz, dtype=float)
xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
# Fields matrix
Ex = np.zeros([Nx, Ny, Nz], dtype=float)
Ey = np.zeros([Nx, Ny, Nz], dtype=float)
Ez = np.zeros([Nx, Ny, Nz], dtype=float)
Bx = np.zeros([Nx, Ny, Nz], dtype=float)
By = np.zeros([Nx, Ny, Nz], dtype=float)
Bz = np.zeros([Nx, Ny, Nz], dtype=float) 
# Fields matrix (old)
Ex_old = np.zeros([Nx, Ny, Nz], dtype=float)
Ey_old = np.zeros([Nx, Ny, Nz], dtype=float)
Ez_old = np.zeros([Nx, Ny, Nz], dtype=float)
Bx_old = np.zeros([Nx, Ny, Nz], dtype=float)
By_old = np.zeros([Nx, Ny, Nz], dtype=float)
Bz_old = np.zeros([Nx, Ny, Nz], dtype=float)
# Increments
dEx = np.zeros([Nx, Ny, Nz], dtype=float)
dEy = np.zeros([Nx, Ny, Nz], dtype=float)
dEz = np.zeros([Nx, Ny, Nz], dtype=float)
dBx = np.zeros([Nx, Ny, Nz], dtype=float)
dBy = np.zeros([Nx, Ny, Nz], dtype=float)
dBz = np.zeros([Nx, Ny, Nz], dtype=float)
# Increments (old)
dEx_old = np.zeros([Nx, Ny, Nz], dtype=float)
dEy_old = np.zeros([Nx, Ny, Nz], dtype=float)
dEz_old = np.zeros([Nx, Ny, Nz], dtype=float)
dBx_old = np.zeros([Nx, Ny, Nz], dtype=float)
dBy_old = np.zeros([Nx, Ny, Nz], dtype=float)
dBz_old = np.zeros([Nx, Ny, Nz], dtype=float)

#Perturbation
Bz[int((Nx - 1)/2), int((Ny - 1)/2), :] = 1.
# Total energy
U = np.zeros(Nt, dtype=float) 
# Divergence of E
divB = np.zeros(Nt, dtype=float) 


# Plot function
def myplot(values, name):
    plt.figure(name)
    plt.imshow(values.T, origin='lower', extent=[0, Lx, 0, Ly], aspect='equal', vmin=-0.1, vmax=0.1, cmap='plasma')
    plt.colorbar()

# Matrix xyz-position shifter with periodic boundary conditions
def shift(mat, x, y, z):
    result = np.roll(mat, -x, 0)
    result = np.roll(result, -y, 1)
    result = np.roll(result, -z, 2)
    return result

# Derivative in x
def ddx(A, s):
    return (shift(A, 1-s, 0, 0) - shift(A, 0-s, 0, 0)) / dx

# Derivative in y
def ddy(A, s):
    return (shift(A, 0, 1-s, 0) - shift(A, 0, 0-s, 0)) / dy

# Derivative in z
def ddz(A, s):
    return (shift(A, 0, 0, 1-s) - shift(A, 0, 0, 0-s)) / dz

# Curl operator
def curl(Ax, Ay, Az, s):
    rx = ddy(Az, s) - ddz(Ay, s)
    ry = ddz(Ax, s) - ddx(Az, s)
    rz = ddx(Ay, s) - ddy(Ax, s)
    return rx, ry, rz

# Divergence operator
def div(Ax, Ay, Az):
    return ddx(Ax, 0) + ddy(Ay, 0) + ddz(Az, 0)


# Time loop
print('Start')
start = time.time()
for t in range(Nt):
    Bx_old[:, :, :] = Bx[:, :, :]
    By_old[:, :, :] = By[:, :, :]
    Bz_old[:, :, :] = Bz[:, :, :]

    dEx, dEy, dEz = curl(Bx, By, Bz, 0)

    Ex += dt*dEx
    Ey += dt*dEy
    Ez += dt*dEz

    dBx, dBy, dBz = curl(Ex, Ey, Ez, 1)

    Bx -= dt*dBx
    By -= dt*dBy
    Bz -= dt*dBz

    U[t] = 0.5*np.sum(Ex**2 + Ey**2 + Ez**2 + Bx*Bx_old + By*By_old + Bz*Bz_old)
    divB[t] = np.sum(np.abs(div(Bx, By, Bz)))
    '''
    if flag_plt == False:
        plt.figure(figsize =(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.pcolor(xv, yv, Bz)
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.plot(divB)
    plt.subplot(2, 2, 3)
    plt.plot(U)
    plt.subplot(2, 2, 4)
    plt.plot(U)
    plt.pause(0.000000000001)

    if flag_start == False:
       command = input("Press Enter, then start.")
       flag_start = True
        
    plt.pause(0.001)
    plt.clf()
    flag_plt=True
    '''
 
    stop = time.time()
    
print('Done!')
print('time = %1.2f s' % (stop - start))


myplot(Ex[:,:,0], 'Ex')
myplot(Ey[:,:,0], 'Ey')
myplot(Bz[:,:,0], 'Bz')

plt.show()
