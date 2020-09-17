import numpy as np
from scipy.optimize import newton_krylov
import matplotlib.pyplot as plt
from scipy import special # special function
from numpy import cosh, zeros_like, mgrid, zeros, ones
# "linspace" return evenly spaced numbers over a specified interval
# "mgrid" returns a dense multi-dimensional mesh grid
# ”zeros_like” return an array of zeros with the same shape and type as a given array
# "zeros" return a new array of given shape and type, filled with zeros
# "ones" return a new array of given shape and type, filled with ones


global source

# simulation parameters
Lx, Ly = 1, 1
nx, ny = 35, 35  
hx, hy = Lx/(nx), Ly/(ny+1)
print("Domain length:", Lx, Ly)
print("Number of vertices:", nx, ny)
print("Increments:", hx, hy)

# boundary condition
P_left, P_right = 0, 0
P_top, P_bottom = 0, 0

# set meshgrid
x = np.linspace(0, Lx, nx)
y = np.linspace(hy, Ly - hy, ny)
xv, yv = np.meshgrid(x, y, indexing = 'ij')
#xv, yv = mgrid[0 : Lx : (nx * 1j), hy : Ly - hy : (ny * 1j)]

def d1nc(N):
    d1x = zeros_like(N)
    d1x[:-1] = np.diff(N, axis = 0)/hx
    d1x[:-1] *= (xv[:-1] + xv[1:])/2.0
    #d1x[-1] = 0
    return d1x

def d1cn(C):
    d1x = zeros_like(C)
    d1x[1:] = np.diff(C, axis = 0)/hx
    d1x[1:] /= xv[1:]
    # Apply Neumann Conditions
    d1x[0] = 0
    return d1x

def d2nn(N,N_right):
    d1x = zeros_like(N)
    d1x[:-1] = np.diff(N, axis = 0)/hx
    d1x[-1] = (N_right - N[-1])/hx
    d1x[:-1] *= (xv[:-1] + xv[1:])/2.0
    
    d2x = zeros_like(d1x)
    d2x[1:] = np.diff(d1x, axis = 0)/hx
    d2x[1:] /= xv[1:]
    # Apply Axis
    d2x[0] = 4 * (N[1] - N[0])/hx/hx
    return d2x

def residual(P):
    d1x = zeros_like(P)
    d2x = zeros_like(P)
    d2y = zeros_like(P)

    #
    #d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2]) / hx/hx
    #d2x[0]    = (P[1]    - 2*P[0]    + P_left)/hx/hx
    #d2x[-1]   = (P_right - 2*P[-1]   + P[-2])/hx/hx

    #d1x = d1nc(P)
    #d2x = d1cn(d1x)
    #

    d2x = d2nn(P, P_right)
 
    d2y[:, 1:-1] = (P[:, 2:] - 2*P[:, 1:-1]  + P[:, :-2])/hy/hy
    d2y[:, 0]    = (P[:, 1]  - 2*P[:, 0]     + P_bottom)/hy/hy
    d2y[:, -1]   = (P_top    - 2*P[:, -1]    + P[:, -2])/hy/hy

    return d2x + d2y  + source

def laplacian(P):
    d1x = zeros_like(P)
    d2x = zeros_like(P)
    d2y = zeros_like(P)

    #
    #d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2])/hx/hx
    #d2x[0]    = (P[1]    - 2*P[0]    + P_left)/hx/hx
    #d2x[-1]   = (P_right - 2*P[-1]   + P[-2])/hx/hx

    #d1x = d1nc(P)
    #d2x = d1cn(d1x)
    #d2x = d2nn(P)
    #

    d1x[:-1] = np.diff(P, axis = 0)/hx
    d1x[:-1] *= (xv[:-1] + xv[1:])/2.0
    #d1x[-1] = 0
    d2x = zeros_like(d1x)
    d2x[1:] = np.diff(d1x, axis = 0)/hx
    d2x[1:] /= xv[1:]
    
    # Apply Axis
    d2x[0] = 4 * (P[1] - P[0])/hx/hx
    d2x = d2nn(P)
    d2y[:, 0]    = (P[:, 1]  - 2 * P[:, 0]    + P_bottom)/hy/hy
    d2y[:, -1]   = (P_top    - 2 * P[:, -1]   + P[:, -2])/hy/hy

    return d2x + d2y 


# solve
kth_zero = special.jn_zeros(0, 1)[-1]
print("kth_zero:", kth_zero)
Bg = pow(np.pi/Ly, 2) + pow(kth_zero/Lx, 2)
source = np.sin(np.pi * yv/Ly) * special.jv(0, kth_zero * xv/Lx)
guess = zeros((nx, ny), float)
sol = newton_krylov(residual, guess, method='lgmres', verbose=1)

print('Residual: %g' % abs(residual(sol)).max())
print('Bg = ', np.max(source)/Bg, '  Numerical = ', np.max(sol))


# visualize
plt.figure(0)
plt.pcolor(xv, yv, sol)
plt.colorbar()

plt.figure(1)
#plt.plot(source.T/Bg/sol)
plt.pcolor(xv, yv, source/Bg)
plt.colorbar()

#plt.figure(2)
#plt.plot(x, sol)
#plt.figure(3)
#plt.plot(x, source/Bg)

plt.show()
