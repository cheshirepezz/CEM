# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:39:40 2017
@author: Owner
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May  7 20:08:42 2017
@author: Owner
"""
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

#initial condition
t = 0.

#const
c = 3.000 * 10**8
eps_0 = 8.854 * 10**(-12)
mu_0 = 1/(c**2 * eps_0)

#parameter
n_co = 3.6
n_cl = 3.24
w = 0.3 * 10**(-6)
n_H = 1.2
n_L = 1.
lambda_0 = 6.0 *10**(-6)
omega_0 = 2.*math.pi*c/lambda_0
area_x = 30.0 * 10**(-6) # and -50.0 um
area_y = 30.0 * 10**(-6) # and -50.0 um
area_z = 30.0 * 10**(-6) # and 100.0 um
t_max = 1.0 * 10**(-9)
m = 0.8
R_0 = 0.01 *10**(-2)
#eps = eps_0
mu = mu_0 #Not magnetic body

#mesh & PML
n_meshx = 30
n_meshy = 30
n_meshz = 90
n_w = 5
n_w2= 30
d_PMLx = n_meshx/10.
d_PMLy = n_meshy/10.
d_PMLz = n_meshz/10.
#Start & End
xs = 1
xe = n_meshx - 1
ys = 1
ye = n_meshy - 1
zs = 1
ze = n_meshz - 1

#diff_table & eps_table
diff_table = np.zeros([n_meshx, n_meshy, n_meshz])
diff_table[0:n_meshx, 0:n_meshy, 0:n_meshz-n_w2] = n_L 
diff_table[0:n_meshx, 0:n_meshy, n_meshz-n_w2:n_meshz] = n_H
eps = np.zeros([n_meshx, n_meshy, n_meshz])
eps[0:n_meshx, 0:n_meshy, 0:n_meshz] = np.power(diff_table[0:n_meshx, 0:n_meshy, 0:n_meshz], 2) * eps_0       

#dt = 1.0 * 10**(-18)
dx = 2*area_x/n_meshx
dy = 2*area_y/n_meshy
dz = 2*area_z/n_meshz 

#for display
H_ = np.zeros([n_meshx, n_meshy, n_meshz])
E_max = []
H_max = []
Exy_for_plt = np.zeros([n_meshx, n_meshy])
Exz_for_plt = np.zeros([n_meshx, n_meshz])
Eyz_for_plt = np.zeros([n_meshy, n_meshz])

#flag
flag_plt = False
flag_start = False
                   
#CFL condition
#v*dt <= 1./sqrt((1/dx)^2+(1/dy)^2+(1/dz)^2)
n_min = min([n_co, n_cl, n_H, n_L])
v_max = c/n_min
dt = 1/5.*1/math.sqrt((1/dx)**2+(1/dy)**2+(1/dz)**2)/v_max

if v_max * dt <= 1/math.sqrt((1/dx)**2+(1/dy)**2+(1/dz)**2):
    pass
else:
    print("Error: Parametars don't meet the CFL condition.")
    command = input("Shall we exit? (Y/N) > ")    
    print(command)
    if command == ("Y") or command == ("y"):
        sys.exit()
    else:
        pass    

#Structing E&H matrix
E_0 = np.zeros([n_meshx, n_meshy, n_meshz])
H_0 = np.zeros([n_meshx, n_meshy, n_meshz])

#Structing PML
"""
sigma_maxx = -(m+1)*math.ln(R_0)/(2*eta_0*eps_0*d_PMLx) # It shoud be modified later.
for xi_x in d_PMLx:
    
    sigma = (xi_x/d_PMLx)**m * sigma_maxx
"""    

print("Initializing Done.")    

"""
Yee Grid (Staggered grid)
E[n-1] H[n-1/2] E[n] H[n+1/2] E[n+1]
E[0               1             2     ]
H[       0             1              ]
E_n[t, x, y, z]
t= n*dt
E_upd = f(E)
"""
E_nx = H_0
E_ny = H_0
E_nz = H_0
H_nx = E_0
H_ny = E_0
H_nz = E_0
E_nx1 = np.zeros([n_meshx, n_meshy, n_meshz])
E_ny1 = np.zeros([n_meshx, n_meshy, n_meshz])
E_nz1 = np.zeros([n_meshx, n_meshy, n_meshz])
H_nx1 = np.zeros([n_meshx, n_meshy, n_meshz])
H_ny1 = np.zeros([n_meshx, n_meshy, n_meshz])
H_nz1 = np.zeros([n_meshx, n_meshy, n_meshz])


#for i in range(20, 30):
#    E_nx[i,25,25] = 1
#    E_nx[i,24,25] = 1
#    H_ny1[i+1,25,25] = 0.5
#    H_ny1[i+1,24,25] = 0.5
#E_nx[25,25,25] = 1.

#E_nx[xs:xe, ys:ye, zs:ze] = 1.
#H_ny[25,25,25]=E_nx[25,25,25] /pow(mu/eps, 1/2.)
#Gaussian
#for i in range(1, n_meshx-1): #Calc new E
#    for j in range(1, n_meshy-1):
#        for k in range(1, n_meshz-1):  
#            xc = n_meshx/2
#            yc = n_meshy/2
#            zc = n_meshz/2
#            x = i 
#            y = j
#            z = k
#            r2 = (x - xc)**2 + (y - yc)**2 + (z - zc)**2
#            w2 = 10.
#            #E_nx[i, j, k] = math.exp(-r2 / w2)
#            E_nx[i, j, k] = math.exp(-((x - xc)**2 + (y - yc)**2 + (z - zc)**2) / w2)
#            E_ny[i, j, k] = math.exp(-((x - xc)**2 + (y - yc)**2 + (z - zc)**2) / w2)
#            E_nz[i, j, k] = math.exp(-((x - xc)**2 + (y - yc)**2 + (z - zc)**2) / w2)
#    

#Difference method (FDTD)
for t in range(1,int(t_max/dt)-1):
    print("n = "+ str(t))
    print("t = "+ str(round(t*dt*10**15,3)) + " [fs]")
    
    #Forced Oscillation
#    for i in range(n_w, n_meshx-1-n_w): #Calc new E
#          for j in range(n_w, n_meshy-1-n_w):
#              E_nx[i, j, int(n_meshz/2.)] = math.sin(omega_0*t*dt)
    E_nx[n_w:xe-n_w, n_w:ye-n_w, int(n_meshz/2.)] = np.sin(omega_0*t*dt)
    
    E_nx1[xs:xe, ys:ye, zs:ze] = E_nx[xs:xe, ys:ye, zs:ze] \
             + dt / (eps[xs:xe, ys:ye, zs:ze]*dy) * (H_nz[xs:xe, ys:ye, zs:ze] - H_nz[xs:xe, ys-1:ye-1, zs:ze]) \
             - dt / (eps[xs:xe, ys:ye, zs:ze]*dz) * (H_ny[xs:xe, ys:ye, zs:ze] - H_ny[xs:xe, ys:ye, zs-1:ze-1]) 
             
    E_ny1[xs:xe, ys:ye, zs:ze] = E_ny[xs:xe, ys:ye, zs:ze] \
             + dt / (eps[xs:xe, ys:ye, zs:ze]*dz) * (H_nx[xs:xe, ys:ye, zs:ze] - H_nx[xs:xe, ys:ye, zs-1:ze-1]) \
             - dt / (eps[xs:xe, ys:ye, zs:ze]*dx) * (H_nz[xs:xe, ys:ye, zs:ze] - H_nz[xs-1:xe-1, ys:ye, zs:ze]) 
             
    E_nz1[xs:xe, ys:ye, zs:ze] = E_nz[xs:xe, ys:ye, zs:ze] \
             + dt / (eps[xs:xe, ys:ye, zs:ze]*dx) * (H_ny[xs:xe, ys:ye, zs:ze] - H_ny[xs-1:xe-1, ys:ye, zs:ze]) \
             - dt / (eps[xs:xe, ys:ye, zs:ze]*dy) * (H_nx[xs:xe, ys:ye, zs:ze] - H_nx[xs:xe, ys-1:ye-1, zs:ze])       
    
    #Boundary Condition (Mur 1) for E-field
    E_nx1[0, ys:ye, zs:ze] = E_nx[1, ys:ye, zs:ze] +(c*dt-dx)/(c*dt+dx)*(E_nx1[1, ys:ye, zs:ze] - E_nx[0, ys:ye, zs:ze])
    E_ny1[0, ys:ye, zs:ze] = E_ny[1, ys:ye, zs:ze] +(c*dt-dx)/(c*dt+dx)*(E_ny1[1, ys:ye, zs:ze] - E_ny[0, ys:ye, zs:ze])
    E_nz1[0, ys:ye, zs:ze] = E_nz[1, ys:ye, zs:ze] +(c*dt-dx)/(c*dt+dx)*(E_nz1[1, ys:ye, zs:ze] - E_nz[0, ys:ye, zs:ze])
    E_nx1[n_meshx-1, ys:ye, zs:ze] = E_nx[n_meshx-2, ys:ye, zs:ze] +(c*dt-dx)/(c*dt+dx)*(E_nx1[n_meshx-2, ys:ye, zs:ze] - E_nx[n_meshx-1, ys:ye, zs:ze])
    E_ny1[n_meshx-1, ys:ye, zs:ze] = E_ny[n_meshx-2, ys:ye, zs:ze] +(c*dt-dx)/(c*dt+dx)*(E_ny1[n_meshx-2, ys:ye, zs:ze] - E_ny[n_meshx-1, ys:ye, zs:ze])
    E_nz1[n_meshx-1, ys:ye, zs:ze] = E_nz[n_meshx-2, ys:ye, zs:ze] +(c*dt-dx)/(c*dt+dx)*(E_nz1[n_meshx-2, ys:ye, zs:ze] - E_nz[n_meshx-1, ys:ye, zs:ze])
      
    E_nx1[xs:xe, 0, zs:ze] = E_nx[xs:xe, 1, zs:ze] +(c*dt-dy)/(c*dt+dy)*(E_nx1[xs:xe, 1, zs:ze] - E_nx[0, ys:ye, zs:ze])
    E_ny1[xs:xe, 0, zs:ze] = E_ny[xs:xe, 1, zs:ze] +(c*dt-dy)/(c*dt+dy)*(E_ny1[xs:xe, 1, zs:ze] - E_ny[0, ys:ye, zs:ze])
    E_nz1[xs:xe, 0, zs:ze] = E_nz[xs:xe, 1, zs:ze] +(c*dt-dy)/(c*dt+dy)*(E_nz1[xs:xe, 1, zs:ze] - E_nz[0, ys:ye, zs:ze])
    E_nx1[xs:xe, n_meshy-1, zs:ze] = E_nx[xs:xe, n_meshy-2, zs:ze] +(c*dt-dy)/(c*dt+dy)*(E_nx1[xs:xe, n_meshy-2, zs:ze] - E_nx[n_meshy-1, ys:ye, zs:ze])
    E_ny1[xs:xe, n_meshy-1, zs:ze] = E_ny[xs:xe, n_meshy-2, zs:ze] +(c*dt-dy)/(c*dt+dy)*(E_ny1[xs:xe, n_meshy-2, zs:ze] - E_ny[n_meshy-1, ys:ye, zs:ze])
    E_nz1[xs:xe, n_meshy-1, zs:ze] = E_nz[xs:xe, n_meshy-2, zs:ze] +(c*dt-dy)/(c*dt+dy)*(E_nz1[xs:xe, n_meshy-2, zs:ze] - E_nz[n_meshy-1, ys:ye, zs:ze])

    E_nx1[xs:xe, ys:ye, 0] = E_nx[xs:xe, ys:ye, 1] +(c*dt-dz)/(c*dt+dz)*(E_nx1[xs:xe, ys:ye, 1] - E_nx[xs:xe, ys:ye, 0])
    E_ny1[xs:xe, ys:ye, 0] = E_ny[xs:xe, ys:ye, 1] +(c*dt-dz)/(c*dt+dz)*(E_ny1[xs:xe, ys:ye, 1] - E_ny[xs:xe, ys:ye, 0])
    E_nz1[xs:xe, ys:ye, 0] = E_nz[xs:xe, ys:ye, 1] +(c*dt-dz)/(c*dt+dz)*(E_nz1[xs:xe, ys:ye, 1] - E_nz[xs:xe, ys:ye, 0])
    E_nx1[xs:xe, ys:ye, n_meshz-1] = E_nx[xs:xe, ys:ye, n_meshz-2] +(c*dt-dz)/(c*dt+dz)*(E_nx1[xs:xe, ys:ye, n_meshz-2] - E_nx[xs:xe, ys:ye, n_meshy-1])
    E_ny1[xs:xe, ys:ye, n_meshz-1] = E_ny[xs:xe, ys:ye, n_meshz-2] +(c*dt-dz)/(c*dt+dz)*(E_ny1[xs:xe, ys:ye, n_meshz-2] - E_ny[xs:xe, ys:ye, n_meshy-1])
    E_nz1[xs:xe, ys:ye, n_meshz-1] = E_nz[xs:xe, ys:ye, n_meshz-2] +(c*dt-dz)/(c*dt+dz)*(E_nz1[xs:xe, ys:ye, n_meshz-2] - E_nz[xs:xe, ys:ye, n_meshy-1])
        
    ### update E
    E_nx = E_nx1
    E_ny = E_ny1
    E_nz = E_nz1
            
    H_nx1[xs:xe, ys:ye, zs:ze] = H_nx[xs:xe, ys:ye, zs:ze] \
             + dt / (mu*dz) * (E_ny[xs:xe, ys:ye, zs+1:ze+1] - E_ny[xs:xe, ys:ye, zs:ze]) \
             - dt / (mu*dy) * (E_nz[xs:xe, ys+1:ye+1, zs:ze] - E_nz[xs:xe, ys:ye, zs:ze]) 
             
    H_ny1[xs:xe, ys:ye, zs:ze] = H_ny[xs:xe, ys:ye, zs:ze] \
             + dt / (mu*dx) * (E_nz[xs+1:xe+1, ys:ye, zs:ze] - E_nz[xs:xe, ys:ye, zs:ze]) \
             - dt / (mu*dz) * (E_nx[xs:xe, ys:ye, zs+1:ze+1] - E_nx[xs:xe, ys:ye, zs:ze]) 
             
    H_nz1[xs:xe, ys:ye, zs:ze] = H_nz[xs:xe, ys:ye, zs:ze] \
             + dt / (mu*dy) * (E_nx[xs:xe, ys+1:ye+1, zs:ze] - E_nx[xs:xe, ys:ye, zs:ze]) \
             - dt / (mu*dx) * (E_ny[xs+1:xe+1, ys:ye, zs:ze] - E_ny[xs:xe, ys:ye, zs:ze]) 
             
    #Update
    H_nx = H_nx1
    H_ny = H_ny1
    H_nz = H_nz1
    #Plotting
#    for i in range(0, n_meshx-1):
#        for j in range(0, n_meshy-1):
#            Exy_for_plt[i, j] = math.pow(E_nx[i, j, 25]**2. +E_ny[i, j, 25]**2. +E_nz[i, j, 25]**2. , 1./2.)
    Exy_for_plt[0:n_meshx, 0:n_meshy] = np.power( \
               np.power(E_nx[0:n_meshx, 0:n_meshy, int(n_meshz/2.)], 2.) + \
               np.power(E_ny[0:n_meshx, 0:n_meshy, int(n_meshz/2.)], 2.) + \
               np.power(E_nz[0:n_meshx, 0:n_meshy, int(n_meshz/2.)], 2.) , 1./2.) 
#    for i in range(0, n_meshx-1):
#        for k in range(0, n_meshz-1):
#            Exz_for_plt[i, k] = math.pow(E_nx[i, 25, k]**2. +E_ny[i, 25, k]**2. +E_nz[i, 25, k]**2. , 1./2.)
    Exz_for_plt[0:n_meshx, 0:n_meshz] = np.power( \
               np.power(E_nx[0:n_meshx, int(n_meshy/2.), 0:n_meshz], 2.) + \
               np.power(E_ny[0:n_meshx, int(n_meshy/2.), 0:n_meshz], 2.) + \
               np.power(E_nz[0:n_meshx, int(n_meshy/2.), 0:n_meshz], 2.) , 1./2.)
#    for j in range(0, n_meshy-1):
#        for k in range(0, n_meshz-1):
#            Eyz_for_plt[j, k] = math.pow(E_nx[25, j, k]**2. +E_ny[25, j, k]**2. +E_nz[25, j, k]**2. , 1./2.)
    Eyz_for_plt[0:n_meshy, 0:n_meshz] = np.power( \
               np.power(E_nx[int(n_meshx/2.), 0:n_meshy, 0:n_meshz], 2.) + \
               np.power(E_ny[int(n_meshx/2.), 0:n_meshy, 0:n_meshz], 2.) + \
               np.power(E_nz[int(n_meshx/2.), 0:n_meshy, 0:n_meshz], 2.) , 1./2.)   
    E_max.append([t*dt*10**15, max(Exy_for_plt.max(), Exz_for_plt.max(), Eyz_for_plt.max())]) 
    H_[xs:xe, ys:ye, zs:ze] = np.power( np.power(H_nx[xs:xe, ys:ye, zs:ze], 2.) + \
                                          np.power(H_ny[xs:xe, ys:ye, zs:ze], 2.) + \
                                            np.power(H_nz[xs:xe, ys:ye, zs:ze], 2.), 1./2.)  

    H_max.append([t*dt*10**15, np.max(H_)*1000]) 
    print("Emax = "+str(Exy_for_plt.max()))
    
    #Plotting
    x_plt = np.arange(-(area_x)*10**6, area_x*10**6, dx*10**6)
    y_plt = np.arange(-(area_y)*10**6, area_y*10**6, dy*10**6)
    z_plt = np.arange(-(area_z)*10**6, area_z*10**6, dz*10**6)
    E_max_plt = np.array(E_max)
    H_max_plt = np.array(H_max)
     
    Y1, X1 = np.meshgrid(y_plt, x_plt)
    Z, X2 = np.meshgrid(z_plt, x_plt)
    Z, Y = np.meshgrid(z_plt, y_plt)
    
    if flag_plt == False:
        plt.figure(figsize =(12, 10))
    
    plt.subplot(2,2,1)
    plt.pcolor(X1, Y1, Exy_for_plt)
    plt.title('|E (X-Y plane)|')
    plt.xlabel("y")
    plt.ylabel("x")
    plt.colorbar()
    
    plt.subplot(2,2,2)
    plt.pcolor(X2, Z, Exz_for_plt)
    plt.title('|E (X-Z plane)|')
    plt.xlabel("x")
    plt.ylabel("z")
    plt.colorbar()
    
    plt.subplot(2,2,3)
    plt.pcolor(Y, Z, Eyz_for_plt)
    plt.title('|E (Y-Z plane)|')
    plt.xlabel("y")
    plt.ylabel("z")
    plt.colorbar()
    
    plt.subplot(2,2,4)
    plt.plot(E_max_plt[:, 0], E_max_plt[:, 1])
    plt.plot(E_max_plt[:, 0], H_max_plt[:, 1])
    plt.title('Emax (t)')
    plt.xlabel("time")
    plt.ylabel("E")
    
    if flag_start == False:
        command = input("Press Enter, then start.")
        flag_start = True
        
    plt.pause(0.001)
    plt.clf()
    flag_plt=True

#Visualization
"""
E_nx_for_plt=[
E_nx[0,0,0] E_nx[1,0,0] ... E_nx[i,0,0]
E
]
"""
Ex_for_plt = np.zeros([n_meshx, n_meshy, n_meshz])
for i in range(0, n_meshx-1):
    for j in range(0, n_meshy-1):
        Ex_for_plt[i, j] = E_nx[i, j, 0]

x_plt = np.arange(-area_x+dx, area_x-dx, dx)
y_plt = np.arange(-area_y+dx, area_y-dx, dy)

X, Y = np.meshgrid(x_plt, y_plt)
plt.use('Agg')
plt.pcolor(X, Y, Ex_for_plt)
plt.colorbar()
plt.show()