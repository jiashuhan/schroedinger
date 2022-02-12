"""
---------------------------------------------------------------
Numerical Solver for the 1D Time-dependent Schrödinger Equation
---------------------------------------------------------------
"""
import numpy as np, matplotlib.pyplot as plt
from matplotlib import animation
from scipy.special import factorial

class Solver(object):
    def __init__(self, x, psi_x0, V_x, t0=0, kmin=None, hbar=1, m=1):
        """
        Numerical solver for the time-dependent Schrödinger equation

        Parameters
        ----------
        x: array_like, float
            1d array of spatial coordinates of length N

        psi_x0: array_like, complex
            initial wave function at time t0 of length N

        V: array_like, float
            potential as a function of x of length N
        
        t0: float
            initial time
        
        kmin: float
            minimum k value. kmax = 2*pi/dx

        hbar: float
            Planck constant

        m: float
            mass of particle
        """
        self.x = x
        self.psi_x0 = psi_x0
        self.V_x = V_x
        self.t = t0
        self.hbar = hbar
        self.m = m
        self.N = len(x)
        self.dx = x[1]-x[0]
        self.dk = 2*np.pi/(N*self.dx) # smallest k spacing 
        if kmin == None: # set -pi/dx < k < pi/dx
            self.kmin = -np.pi/self.dx # default kmin
        else:
            self.kmin = kmin
        self.k = self.kmin + self.dk*np.arange(self.N)
        # this is the discrete psi_x used for FT; actual psi_x is off by a factor
        self.psi_x = self.dx/np.sqrt(2)*psi_x0*np.exp(-1j*self.kmin*self.x)
        self.x_to_k()
        self.dt = None
        
    def x_to_k(self):
        self.psi_k = np.fft.fft(self.psi_x)

    def k_to_x(self):
        self.psi_x = np.fft.ifft(self.psi_k)

    # returns actual psi_x
    def get_psi_x(self):
        return self.psi_x*np.exp(1j*self.kmin*self.x)*np.sqrt(2*np.pi)/self.dx

    # returns actual psi_k
    def get_psi_k(self):
        return self.psi_k*np.exp(-1j*x[0]*self.dk*np.arange(self.N))

    def evolve(self, dt, N_steps=1):
        """
        Time evolve the wavefunction by dt * N_steps using leap-frog 
        technique. Every single step consists of a half step in x, a 
        full step in k, and then a half step in x

        Parameters
        ----------
        dt: float
            time interval used in integration
        
        N_steps: float
            number of time intervals to evolve
        """
        assert N_steps > 0

        if dt != self.dt:
            self.dt = dt
            self.half_step_x = np.exp(-1j*self.V_x/self.hbar*dt/2)
            self.full_step_x = self.half_step_x * self.half_step_x
            self.full_step_k = np.exp(-1j*self.hbar*self.k**2/2/self.m*self.dt)

        self.psi_x *= self.half_step_x
        
        for i in range(N_steps-1): # full steps in between
            self.x_to_k()
            self.psi_k *= self.full_step_k
            self.k_to_x()
            self.psi_x *= self.full_step_x

        self.x_to_k()
        self.psi_k *= self.full_step_k
        self.k_to_x()
        self.psi_x *= self.half_step_x

        self.t += dt*N_steps

# helper functions for the potential and initial state
def quadratic(x, height_scale):
    return height_scale*x**2

def hermite(x, n):
    c = np.zeros(n+1)
    c[n] = 1
    return np.polynomial.hermite.hermval(x, c)

# m = hbar = 1, omega = 1
def oscillator_state(x, x0, n):
    return np.exp(-(x-x0)**2/2)*hermite(x-x0, n)/np.sqrt(2**n*factorial(n))/np.pi**(1/4)

# step fuction
def theta(x):
    y = np.zeros(x.shape)
    y[x > 0] = 1.0
    return y

def square_barrier(x, width, height):
    return height * (theta(x) - theta(x - width))

# parameters
hbar = 1
m = 1.9
dt = 0.01 # time intervals for integration
N_steps = 50 # N_steps*dt is the time step for each update
t_max = 120 # duration
N_frames = int(t_max/N_steps/dt) # number of frames

# specify x range
N = 2048 # spatial resolution
dx = 0.1
x = dx * (np.arange(N) - 0.5 * N)

# harmonic oscillator potential
scale = 0.01
V_x = quadratic(x, scale)
#V_x += square_barrier(x, 0.1, 70) # add barrier
#V_x[x < -100] = 1e6 # add hard walls on edges
#V_x[x > 100] = 1e6

V0 = 55 # potential value at turning point
x0 = -np.sqrt(V0/scale) # set initial location of wavefunction at left turning point
psi_x0 = oscillator_state(x, x0, 2) # 2nd excited state

# Schrödinger solver for the harmonic potential
solver = Solver(x=x, psi_x0=psi_x0, V_x=V_x, hbar=hbar, m=m, kmin=-20)

# plot animation
fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(6.4,4.8))
plt.subplots_adjust(hspace=0.3)

ymin = np.abs(solver.get_psi_x()).min()
ymax = np.abs(solver.get_psi_x()).max()
psi_x_plot, = ax0.plot([], [], c='r', label='$|\\psi(x)|^2$')
V_x_plot, = ax0.plot([], [], c='k', label='$V(x)$')
title = ax0.set_title("")
ax0.legend(loc='upper right')
ax0.set_xlabel('$x$')
ax0.set_ylabel('$|\\psi(x)|^2$')
ax0.set_xlim(solver.x.min()+0.01*N*dx, solver.x.max()-0.01*N*dx)
ax0.set_ylim(ymin-0.2*(ymax-ymin), ymax+0.2*(ymax-ymin))
V_x_plot.set_data(solver.x, solver.V_x)

ymin = np.abs(solver.get_psi_k()).min()
ymax = np.abs(solver.get_psi_k()).max()
psi_k_plot, = ax1.plot([], [], c='r', label='$|\\psi(k)|^2$')
ax1.legend(loc='upper right')
ax1.set_xlabel('$k$')
ax1.set_ylabel('$|\\psi(k)|^2$')
ax1.set_xlim(solver.k.min(), -solver.k.min())
ax1.set_ylim(ymin-0.2*(ymax-ymin), ymax+1*(ymax-ymin))

def animate(i):
    solver.evolve(dt, N_steps)
    psi_x_plot.set_data(solver.x, np.abs(solver.get_psi_x())**2)
    V_x_plot.set_data(solver.x, solver.V_x/100)
    title.set_text("t = %.2f"%solver.t)
    psi_k_plot.set_data(solver.k, np.abs(solver.get_psi_k())**2)
    return psi_x_plot, V_x_plot, psi_k_plot

anim = animation.FuncAnimation(fig, animate, frames=N_frames, interval=30, blit=False)
#anim.save('./results/oscillator.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
plt.show()
