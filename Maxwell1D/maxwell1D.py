"""A simple 1D Maxwell Solver 
  This one is based on 'maxvell1D.m' matlab routine by G. Lapenta.
  
  (c) V. Olshevsky 2012, mailto: sya@mao.kiev.ua

"""
import numpy as np  ## numeric routines; arrays
import pylab as plt  ## plotting

# Set plotting parameters
params = {'axes.labelsize': 'large',
              'xtick.labelsize': 'medium',
              'ytick.labelsize': 'medium',
              'font.size': 10,
              'text.usetex': False}
plt.rcParams.update(params)
## Switch on interactive plotting mode
#plt.ion()

"""
    dB/dt=-curl E 

    eps0 mu0 dE/dt= curl B - mu0 J 

    ( curl A )_y = - dA_z/dx
    ( curl A )_z =  dA_y/dx    
"""
mu0 = 4. * np.pi * 1e-7
eps0 = 8.8542 * 1.e-12
c = 1. / (mu0 * eps0)**0.5
Lx = 1e6
Time = Lx / c / 4
nx = 1000
dx = Lx / nx
x = np.arange(nx) * dx  #0:dx:Lx-dx
dt = dx / c / 2
Ncycles = int(Time / dt)
k = 2 * np.pi / Lx * 3

"""Choose initial condition: pure wave, wave packet, ot random noise"""
## Pure wave
"""
Ey = np.sin(k * x)
Bz = np.sin(k * (x + dx/2)) / c
"""

## Wave packet

sigma = Lx / 100
Ey = np.exp(-(x - Lx/2)**2 / sigma**2)
Bz = np.zeros(nx)


## Random noise
"""
Ey = np.random.randn(nx)
Bz = np.random.randn(nx)
"""

plt.figure('maxwell1D') 
plt.subplot(2, 2, 1)
plt.plot(x, Ey, 'r', label='Ey')
plt.xlabel('x')
plt.legend(loc=1)
plt.subplot(2, 2, 2)
plt.plot(x+dx/2, Bz, 'b', label='Bz')
plt.xlabel('x')
plt.legend(loc=1)
plt.pause(0.0001)
plt.draw()

curlEz = np.zeros(nx)
curlBy = np.zeros(nx)
# Empty lists for storing the computation results
etm = []
btm = []
t = []
for it in range(1, Ncycles) :
    ## compute central difference, assuming periodicity
    curlEz[0:-1] = (Ey[1:] - Ey[0:-1]) / dx
    curlEz[-1] = (Ey[0] - Ey[len(Ey)-1]) / dx
    Bz = Bz - dt*curlEz
    curlBy[1:] = -(Bz[1:] - Bz[0:-1]) / dx
    curlBy[0] = -(Bz[0] - Bz[-1]) / dx
    Ey = Ey + dt*curlBy/mu0/eps0
    if np.mod(it, Ncycles/10) == 0 :
        plt.clf()
        plt.subplot(2, 2, 1)        
        plt.plot(x, Ey, 'r', label='Ey')        
        plt.title('t=%f' % (it*dt))
        plt.xlabel('x')
        plt.legend(loc=1)        
        plt.subplot(2, 2, 2)
        plt.plot(x+dx/2, Bz, 'b', label='Bz')
        plt.xlabel('x')
        plt.legend(loc=1)
        plt.pause(0.0001)
        plt.draw()
    etm.append(Ey)
    btm.append(Bz)
    t.append(it*dt)

# convert the lists to np.ndarray
etm = np.array(etm)
btm = np.array(btm)
t = np.array(t)

## Plot the variations of electric field
plt.hold(True)
plt.subplot(2, 2, 3)
plt.title('Ey variations')
plt.pcolor(x, t, etm)
plt.xlabel('x')
plt.ylabel('t')
## Plot light cone
plt.plot([Lx/2, Lx/2-c*Time], [0, Time], color='w', linewidth=2)
plt.plot([Lx/2, Lx/2+c*Time], [0, Time], color='w', linewidth=2)
plt.pause(0.0001)
plt.draw()

#plt.figure('Spectra')
etmf2 = np.fft.fft2(etm) # should be fft2

# FFT variable in time
Fs = 1 / dt ## Sampling frequency
T = 1/Fs ## Sample time
Nt = len(t) ## Length of signal
w = Fs * np.pi * np.linspace(0, 1, Nt/2+1)

# FFT variable in space
Fs = 1 / dx ## Sampling frequency
T = 1 / Fs ## Sample length
Nx = len(x) ## Length of signal
k = Fs * np.pi * np.linspace(0, 1, Nx/2+1)

# Plot the spectrum
plt.subplot(2, 2, 4)
plt.title('Spectrum')
plt.pcolor(k, w, abs(etmf2[0:Nt/2+1, 1:Nx/2+1]))
plt.plot(k, k*c, 'w')
plt.xlabel('k')
plt.ylabel('omega')
plt.plot(k, 2*np.arcsin(c * dt/dx * np.sin(k*dx/2))/dt)
plt.pause(0.0001)
plt.draw()

plt.show()
