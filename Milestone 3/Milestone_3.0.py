"""
HPC Milestone 3
Jared Engelken
Python 3
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Data type
dtype = np.float32

# parameter setup
c_x = np.array([0,1,0,-1,0,1,-1,-1,1]) #discretization of the particle velocities in the x direction
c_y = np.array([0,0,1,0,-1,1,1,-1,-1]) #discretization of the particle velocities in the y direction
# c = np.array([[0,  1,  0, -1,  0,  1, -1, -1,  1],    # velocities, x components
#               [0,  0,  1,  0, -1,  1,  1, -1, -1]]).T # velocities, y components
Nx = 15 #plot size in the x direction
Ny = 10 #plot size in the y direction
NC = 9 # number of velocity channels
timestep = 5 #number of animation iterations

weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) # weights for each channel
omega = 0.3
# idxs = np.arange(NC)
tau = 0.6
epsilon = 1

# Probability function setup
grid_size = (Ny,Nx,NC) #form the grid shape: [length y, length x, number of velocities]
f = np.zeros(grid_size) #form the grid initialized with zeros
#f[:,:,0]=1 #modify specific lattice points to represent various particles in motion 
for i in range(len(weights)):
    f[:,:,i]=weights[i]
#e.g. [2,1,5] is a particle at the (x=1, y=2) position that moves in the northeast direction
f[4,7,0]=f[1,1,0]+0.001

X,Y = np.meshgrid(range(Nx),range(Ny)) #create the plot grid

rho = np.sum(f,axis=2) #calculate the density at each lattice point
#rho0 = rho + epsilon*np.sin((2*np.pi*x)/len(x))

#calculate the velocity components in the x and y directions respectively
vel_x = np.divide(np.sum(f*c_x,2),rho, out=np.zeros_like(np.sum(f*c_x,2)), where=rho!=0) 
vel_y = np.divide(np.sum(f*c_y,2),rho, out=np.zeros_like(np.sum(f*c_y,2)), where=rho!=0)

#This function shifts each particle by 1 lattice point according to its velocity direction in each channel
def streaming(var):
    """ STREAMING """
    var[:,:,1]=np.roll(var[:,:,1],(0,1),axis=(0,1)) #shift all particles with a 1 velocity 1 to the right
    var[:,:,2]=np.roll(var[:,:,2],(1,0),axis=(0,1)) #shift all particles with a 2 velocity 1 up
    var[:,:,3]=np.roll(var[:,:,3],(0,-1),axis=(0,1)) #...
    var[:,:,4]=np.roll(var[:,:,4],(-1,0),axis=(0,1))
    var[:,:,5]=np.roll(var[:,:,5],(1,1),axis=(0,1))
    var[:,:,6]=np.roll(var[:,:,6],(-1,1),axis=(0,1))
    var[:,:,7]=np.roll(var[:,:,7],(-1,-1),axis=(0,1))
    var[:,:,8]=np.roll(var[:,:,8],(1,-1),axis=(0,1))
    #xvel = np.sum(f*c_x,axis=2) #recalculate the x velocity
    #yvel = np.sum(f*c_y,axis=2) #recalculate the y velocity
    return var#, xvel, yvel

Feq = np.zeros(f.shape)
u = np.zeros((15,10))
def collision(var, omega):
    # rho = np.sum(f,axis=2)
    # u = np.zeros(f.shape)
    # for i, cx,cy in zip(idxs,c_x, c_y):
    #     u[:,:,i] = (cx*vel_x + cy*vel_y) 
    # u = np.divide(np.sum(u,axis = 2),rho,out=np.zeros_like(np.sum(f*c_y,2)), where=rho!=0)
    # var += omega*(equilibrium(rho, u) - f)
    # return rho, u
    rho = np.sum(var, axis=0)
    ux = (var[1] - var[3] + var[5] - var[6] - var[7] + var[8])/rho
    uy = (var[2] - var[4] + var[5] + var[6] - var[7] - var[8])/rho
    var += omega*(equilibrium(rho, ux, uy) - var)
    return rho, ux, uy

# def equilibrium(rho, var):
#     for i,cx,cy,w in zip(idxs,c_x,c_y,weights):
#         Feq[:,:,i] = rho*w*(1 + 3*u + 9/2*u**2 - 3/2*(vel_x**2+vel_y**2))
#     var += -(1.0/tau) * (var - Feq) 
#     return Feq

def equilibrium(rho, ux, uy):
    # cu_ikl = np.dot(u.T, c.T.astype(u.dtype)).T
    # uu_kl = np.sum(u**2, axis=0)
    # return (weights*(rho*(1 + 3*cu_ikl + 9/2*cu_ikl**2 - 3/2*uu_kl)).T).T
    c5 = ux + uy
    c6 = -ux + uy
    c7 = -ux - uy
    c8 = ux - uy
    uu = ux**2 + uy**2
    return np.array([weights[0]*rho*(1 - 3/2*uu),
    weights[1]*rho*(1 + 3*ux + 9/2*ux**2 - 3/2*uu),
    weights[2]*rho*(1 + 3*uy + 9/2*uy**2 - 3/2*uu),
    weights[3]*rho*(1 - 3*ux + 9/2*ux**2 - 3/2*uu),
    weights[4]*rho*(1 - 3*uy + 9/2*uy**2 - 3/2*uu),
    weights[5]*rho*(1 + 3*c5 + 9/2*c5**2 - 3/2*uu),
    weights[6]*rho*(1 + 3*c6 + 9/2*c6**2 - 3/2*uu),
    weights[7]*rho*(1 + 3*c7 + 9/2*c7**2 - 3/2*uu),
    weights[8]*rho*(1 + 3*c8 + 9/2*c8**2 - 3/2*uu)])

### Initialize probability distribution with a shear wave
x_k = np.arange(Nx)
wavevector = 2*np.pi/Nx
uy_k = np.sin(wavevector*x_k, dtype=dtype)
u_ck = np.array([np.zeros_like(uy_k), uy_k], dtype=dtype)

f = equilibrium(np.ones((Nx, Ny), dtype=dtype), np.zeros_like(uy_k).reshape((Nx, 1)), uy_k.reshape((Nx, 1)))

# for i in range(timestep): #plot the particle movement for the number of given timesteps
#     plt.subplots()
#     plt.axis('equal')
#     plt.quiver(X, Y, vel_x, vel_y, scale=1, units='xy')
#     f,vel_x,vel_y,rho=streaming(f) #shift the particles after each plot
#     u = collision(f)
#     Feq = equilibrium(f)
#     cmatrix = plt.imshow(rho)
#     plt.colorbar(cmatrix)
# plt.show()

amplitude = []
for i in range(timestep):
    streaming(f)
    rho, ux, uy = collision(f, omega)
    # Fourier analysis
    amplitude += [(uy[:, Ny//2]*uy_k).sum() * 2/Nx]

print(amplitude)

