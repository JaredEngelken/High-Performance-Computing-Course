"""
HPC Milestone 4
Jared Engelken
Python 3
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from enum import IntEnum


### Set Parameters, define grid & set initialization values
dtype = np.float32 #data type
timestep = 100 #number of animation iterations
Nx = 30 #plot size in the x direction #15
Ny = 30 #plot size in the y direction #10
Y,X = np.meshgrid(range(Ny),range(Nx)) #create the plot grid

c = np.array([[0,  1,  0, -1,  0,  1, -1, -1,  1],    # velocities, x components
              [0,  0,  1,  0, -1,  1,  1, -1, -1]]).T # velocities, y components
weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) # weights for each channel
omega = 0.5  #relaxation rate #0.3 #A higher omega means a lower kinematic viscosity. Anything above 1.7 becomes unstable!
epsilon = 0.001 #pressure difference
wall_vel = 0.1 #wall speed

# Don't show any plots
plt.ioff()

# Define starting functions & values for u & rho
x_k = np.arange(Nx)
wavevector = 2*np.pi/Nx
uy_k = np.sin(wavevector*x_k, dtype=dtype)

rho = np.ones((Nx,Ny))    

ux = np.zeros_like(uy_k).reshape((Nx, 1))
u0 = np.zeros_like(uy_k).reshape((Nx, 1))
uy = uy_k.reshape((Nx, 1))

# Directions
D = IntEnum('D', 'E N W S NE NW SW SE')

### Compute functions
def equilibrium(rho_kl, ux_kl, uy_kl):
    cu5_kl = ux_kl + uy_kl
    cu6_kl = -ux_kl + uy_kl
    cu7_kl = -ux_kl - uy_kl
    cu8_kl = ux_kl - uy_kl
    uu_kl = ux_kl**2 + uy_kl**2
    return np.array([weights[0]*rho_kl*(1 - 3/2*uu_kl),
                     weights[1]*rho_kl*(1 + 3*ux_kl + 9/2*ux_kl**2 - 3/2*uu_kl),
                     weights[2]*rho_kl*(1 + 3*uy_kl + 9/2*uy_kl**2 - 3/2*uu_kl),
                     weights[3]*rho_kl*(1 - 3*ux_kl + 9/2*ux_kl**2 - 3/2*uu_kl),
                     weights[4]*rho_kl*(1 - 3*uy_kl + 9/2*uy_kl**2 - 3/2*uu_kl),
                     weights[5]*rho_kl*(1 + 3*cu5_kl + 9/2*cu5_kl**2 - 3/2*uu_kl),
                     weights[6]*rho_kl*(1 + 3*cu6_kl + 9/2*cu6_kl**2 - 3/2*uu_kl),
                     weights[7]*rho_kl*(1 + 3*cu7_kl + 9/2*cu7_kl**2 - 3/2*uu_kl),
                     weights[8]*rho_kl*(1 + 3*cu8_kl + 9/2*cu8_kl**2 - 3/2*uu_kl)])

def collision(f, omega):
    rho_kl = np.sum(f, axis=0)
    ux_kl = (f[1] - f[3] + f[5] - f[6] - f[7] + f[8])/rho_kl
    uy_kl = (f[2] - f[4] + f[5] + f[6] - f[7] - f[8])/rho_kl
    f += omega*(equilibrium(rho_kl, ux_kl, uy_kl) - f)
    return rho_kl, ux_kl, uy_kl

def streaming(f):
    for i in range(1, 9):
        f[i] = np.roll(f[i], c[i], axis=(0, 1))

def bounce_back (f, wall_vel):
    f_bottom = f[:,:,0].copy()
    f_top = f[:,:,-1].copy()    
    f_left = f[:,0,:].copy()
    f_right = f[:,-1,:].copy()

    streaming(f)
    # Defining the boundary conditions --> Anti Direction for hitting the bottom wall
    f[2,:,0] = f_bottom[4]                      #North -> South
    f[5,:,0] = f_bottom[7]                      #North-East -> South-West
    f[6,:,0] = f_bottom[8]                      #North_West -> South-East
    
    
    # Top boundary: sliding lid - compute rho after bounce back
    rho_bottom = f_top[6] + f_top[2] + f_top[5] + f[6,:,-1] + f[2,:,-1] + f[5,:,-1] + f[3,:,-1] + f[0,:,-1] + f[1,:,-1]
    f[4,:,-1] = f_top[4]
    f[8,:,-1] = f_top[6] + 6*weights[8]*rho_bottom*wall_vel
    f[7,:,-1] = f_top[5] - 6*weights[7]*rho_bottom*wall_vel
    
     # Change to "if False" for Couette flow test, otherwise set to "if True"
    if False:
        # Left boundary
        f[D.E, 0, :] = f_left[D.W]
        f[D.NE, 0, :] = f_left[D.SW]
        f[D.SE, 0, :] = f_left[D.NW]

        # Right boundary
        f[D.W, -1, :] = f_right[D.E]
        f[D.NW, -1, :] = f_right[D.SE]
        f[D.SW, -1, :] = f_right[D.NE]

        # Bottom-left corner
        f[D.N, 0, 0] = f_bottom[D.S, 0]
        f[D.E, 0, 0] = f_bottom[D.W, 0]
        f[D.NE, 0, 0] = f_bottom[D.SW, 0]

        # Bottom-right corner
        f[D.N, -1, 0] = f_bottom[D.S, -1]
        f[D.W, -1, 0] = f_bottom[D.E, -1]
        f[D.NW, -1, 0] = f_bottom[D.SE, -1]

        # Top-left corner
        f[D.S, 0, -1] = f_top[D.N, 0]
        f[D.E, 0, -1] = f_top[D.W, 0]
        f[D.SE, 0, -1] = f_top[D.NW, 0]

        # Top-right corner
        f[D.S, -1, -1] = f_top[D.N, -1]
        f[D.W, -1, -1] = f_top[D.E, -1]
        f[D.SW, -1, -1] = f_top[D.NE, -1]
     

### Calculation
# Definition of the probability density function of via calling equilibrium function
# First Roadpoint - Set u = 0 & rho = p0 + epsilon (2pix/lx)
f = equilibrium(rho, ux, u0)

### Quiver Plot
for k in range(timestep):
    bounce_back(f, wall_vel)
    rho_kl, ux_kl, uy_kl = collision(f, omega)
    
    if k%10==0:
        # Plot results
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.text(Nx/2, Ny+1, 'Moving Wall', c = 'red', horizontalalignment = 'center')
        ax.text(Nx + 1, Ny/2, 'Outlet', c = 'blue',  verticalalignment='center')
        ax.text(0-6, Ny/2, 'Inlet', c = 'blue',  verticalalignment='center')
        ax.text(Nx/2, 0 - 5, 'Bottom Wall', c = 'blue', horizontalalignment = 'center')        
        ### Now this loop is plotting the quiver function instead of the density. Due to lazyness 
        ### however, i am to lazy, to make this in a new loop right now. To see the density, just
        ### activate all ha. lines in the code and comment the he. lines
        # ax = fig.add_subplot(111,projection = '3d')
        # ax.set_zlim([0.95,1.1])
        # ax.set_xlabel('x-Axis (Point in Grid')
        # ax.set_ylabel('y-Axis (Point in Grid)')
        # ax.set_zlabel('z-Axis (Density)')
        ax.quiver(X,Y,ux_kl,uy_kl)
        plt.savefig('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images/Figures_2_Density' + str(k).zfill(4))


# Definition of the probability density function of via calling equilibrium function
# # First Roadpoint - Set u = 0 & rho = p0 + epsilon (2pix/lx)
f = equilibrium(rho, ux, u0)

### Surface Plot
for k in range(timestep):
    bounce_back(f, wall_vel)
    rho_kl, ux_kl, uy_kl = collision(f, omega)
    
    if k%10==0:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111,projection = '3d')
        # ax1.set_zlim([-0.1,0.1])
        ax1.set_xlabel('x-Axis (Point in Grid')
        ax1.set_ylabel('y-Axis (Point in Grid)')
        ax1.set_zlabel('z-Axis (Velocity)')
        ax1.plot_surface(X,Y,ux_kl)
        plt.savefig('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images/Figures_2_Velocity' + str(k).zfill(4))

