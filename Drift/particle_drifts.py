#! /usr/bin/python
# 
#  Python script for computing and plotting single charged particle 
#  trajectories in prescribed electric and magnetic fields.
#  Loosely based on boris.m matlab program

#  Accompanies Plasma Dynamics lecture on single particle dynamics

#  Paul Gibbon, October 2015
#  Last modified: Oct 2018

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import griddata
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import os
import os.path
import sys
from time import sleep


def integrate(E0, B0, vz0):
   global dt, v0, x0, xp, yp, zp, qom, larmor, nsteps
   wc=qom*B0 # cyclotron frequency
   larmor=vperp/wc
   print ("Cyclotron frequency =",wc)
   print ("Perpendicular velocity v_p=",vperp)
   print ("Larmor radius=",larmor)
   norm = 1.e-6  # choose whether to normalise plot axes dimensions to Larmor radius
#   norm = larmor
   trun=5*2*np.pi/wc  # total runtime
   dt=.1/wc  # timestep - adjust to current B-field

   nsteps=int(trun/dt)  # timesteps
   E=np.array([0.,E0,0.])  # initial E-field
   B=np.array([0.,0.,B0])  # initial B-field
   u=np.array([0.,0.,0.])  # intermediate velocity
   h=np.array([0.,0.,0.])  # normalized B-field
   xp[0]=x0[0]
   yp[0]=x0[1]
   zp[0]=x0[2]
   v0[2]=vz0 # z-component

   v=v0+.5*dt*qom*(E+np.cross(v0,B)) # shift initial velocity back 1/2 step
   x=x0

   for itime in range(1,nsteps):
     x=x+dt*v
     xp[itime]=x[0] /norm
     yp[itime]=x[1] /norm
     zp[itime]=x[2] /norm
#
# Boris mover: solves dv/dt = q/m*(E + vxB) to 2nd order accuracy in dt
#
     qomdt2 = dt*qom/2
     h = qomdt2*B
     s=2*h/(1+np.dot(h,h)) 
     u = v + qomdt2*E
     up=u+np.cross(u+np.cross(u,h),s)
     v=up+qomdt2*E
  
# ===================================
     
# Make 3D plot of particle orbit

def plot_track():
  global xp,yp,zp,nsteps,ax1
  xmin=np.min(xp)
  xmax=np.max(xp)
  ymin=np.min(yp)
  ymax=np.max(yp)
  zmin=np.min(zp)
  zmax=np.max(zp)
  ax1.cla()

  plt.ion()
  plt.grid(True, which='both')
  ax1.set_xlim( (xmin, xmax) )
  ax1.set_ylim( (ymin, ymax) )
  ax1.set_zlim( (zmin, zmax) )
  ax1.set_xlabel('$x $[microns]')
  ax1.set_ylabel('$y $[microns]')
  ax1.set_zlabel('$z $[microns]')
#ax1.set_aspect(1.)
  ax1.scatter(xp[0:nsteps],yp[0:nsteps],zp[0:nsteps],c='r',marker='o')
  plt.draw()

# =============================================
#
#  Main program

print ("Charged particle orbit solver")
plotboxsize   = 8.
animated = True

fig = plt.figure(figsize=(8,8))
#fig.suptitle("Tracks")

x0=np.array([0.,0.,0.])     # initial coords
vz0=0.
v0=np.array([-1e2,0.,vz0]) # initial velocity
vperp = np.sqrt(v0[0]**2+v0[2]**2)
E0=0.
B0=.1

e=1.602176e-19 # electron charge
m=9.109e-31 # electron mass
qom=e/m  # charge/mass ratio

wc=qom*B0 # cyclotron frequency
larmor=vperp/wc
print (wc,vperp,larmor)

trun=5*2*np.pi/wc  # total runtime
dt=.1/wc  # timestep - adjust to current B-field

nsteps=int(trun/dt)  # timesteps
B1=np.array([0.,0.,0.1])  # gradient B perturbation

#wc=qom*np.linalg.norm(B) # cyclotron frequency

#nsteps=2
xp = np.zeros(nsteps)  # particle tracks
yp = np.zeros(nsteps) #
zp = np.zeros(nsteps)

integrate(E0, B0, vz0)
  # Get instance of Axis3D
ax1 = fig.add_subplot(111, projection='3d')
  
# Get current rotation angle
print (ax1.azim)
 
# Set initial view to x-y plane
ax1.view_init(elev=90,azim=0)
ax1.set_xlabel('$x $[microns]')
ax1.set_ylabel('$y $[microns]')
ax1.set_zlabel('$z $[microns]')
plot_track()

#filename = 'a0_45/parts_p0000.%0*d'%(6, ts)
#plot_from_file(filename):
axcolor = 'lightgoldenrodyellow'
axe0 = fig.add_axes([0.1, 0.95, 0.3, 0.03], facecolor=axcolor) # box position, color & size
axb0  = fig.add_axes([0.5, 0.95, 0.3, 0.03], facecolor=axcolor)
axv0  = fig.add_axes([0.1, 0.9, 0.3, 0.03], facecolor=axcolor)

sefield = Slider(axe0, 'Ey [V/m]', 0.0,10.0, valinit=E0)
sbfield = Slider(axb0, 'Bz [T]', 0.0, 1.0, valinit=B0)
svz = Slider(axv0, 'vz [m/s]', 0.0, 1.0, valinit=0.)

def update(val):
    E0 = sefield.val
    B0 = sbfield.val
    vz0 = svz.val

    integrate(E0,B0,vz0)
    plot_track()
    plt.draw()
    
sefield.on_changed(update)
sbfield.on_changed(update)
svz.on_changed(update)

       
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    global ax1
    sefield.reset()
    sbfield.reset()
    svz.reset()
    ax1.cla()
    ax1.set_xlabel('$x $[microns]')
    ax1.set_ylabel('$y $[microns]')
    ax1.set_xlim( (0., 10.) )
#    ax1.set_ylim( (-sigma, sigma) )
    ax1.grid(True, which='both')
    plt.draw()
button.on_clicked(reset)

       
plt.show()
