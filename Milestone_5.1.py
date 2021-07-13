"""
HPC Milestone 5
Jared Engelken
Python 3
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from enum import IntEnum
import matplotlib.animation as animation
import time


### Set Parameters, define grid & set initialization values
dtype = np.float32 #data type
timestep = 10000 #number of animation iterations
Nx = 30 #plot size in the x direction #15
Ny = 30 #plot size in the y direction #10
x = np.arange(Nx)+0.5
y = np.arange(Ny)+0.5
X,Y = np.meshgrid(x,y) #create the plot grid

c = np.array([[0,  1,  0, -1,  0,  1, -1, -1,  1],    # velocities, x components
              [0,  0,  1,  0, -1,  1,  1, -1, -1]]).T # velocities, y components
weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) # weights for each channel
omega = 0.5  #relaxation rate #0.3 #A higher omega means a lower kinematic viscosity. Anything above 1.7 becomes unstable!
epsilon = 0.001 #pressure difference
wall_vel = 0 #0.1 #wall speed
u_max = 2.0 #maximum velocity
nu = (1/omega-0.5)/3 #kinematic shear viscosity
Re = 0.1*np.max([Nx,Ny])/nu #Reynold's number
rho0 = 1 #rest density
pin = rho0+epsilon #density at the pipe start
pout = rho0-epsilon #density at the pipe end

# Define starting functions & values for u & rho
x_k = np.arange(Nx+2) # x_k = np.arange(Nx)
wavevector = 2*np.pi/Nx
uy_k = np.sin(wavevector*x_k, dtype=dtype)
rho = rho0*np.ones((Nx+2,Ny+2))
u = np.zeros((2,Nx+2,Ny+2))

def equilibrium(rho_kl, u_ckl):
    cu_ikl= np.dot(u_ckl.T, c.T).T
    uu_kl= np.sum(u_ckl**2, axis=0)
    return(weights*(rho_kl*(1 + 3*cu_ikl+ 9/2*cu_ikl**2 - 3/2*uu_kl)).T).T # calculate the fluid particle equilibrium distribution

# def collision(f_ikl, omega):
#     rho_kl = np.sum(f_ikl, axis=0) # calculate the density of fluid particles
#     u_ckl = np.dot(f_ikl.T, c).T/rho_kl # create a velocity field with respect to the particle vectors and density -> collisions
#     f_ikl+= omega*(equilibrium(rho_kl, u_ckl)-f_ikl) # recalculate the fluid particle equilibrium distribution
#     return rho_kl, u_ckl

def collision(f, omega):
    rho = np.sum(f, axis=0)
    ux = (f[1] - f[3] + f[5] - f[6] - f[7] + f[8])/rho
    uy = (f[2] - f[4] + f[5] + f[6] - f[7] - f[8])/rho
    u_ckl = np.dot(f.T, c).T/rho
    f += omega*(equilibrium(rho, u_ckl) - f)
    return rho, uy, ux

def streaming(f, c):
    for i in range(1, 9): #for each of the 9 velocity channel directions
        f[i] = np.roll(f[i], c[i], axis=(0, 1)) #shift the fluid particles 1 space in their given direction
    return f

def bounce_back (f):
    ### This code works
    f_bottom = f[:,:,0].copy()
    f_top = f[:,:,-1].copy()
    # # Defining the boundary conditions --> Anti Direction for hitting the top and bottom wall
    f[[2,5,6],:,1] = f_bottom[[4,7,8],:]
    f[4,:,-2] = f_top[2,:]
    f[8,1:-1,-2] = f_top[6,:-2]
    f[7,1:-1,-2] = f_top[5,:-2]

    ### This code below also works
    # f[[2,5,6],:,1] = f[[4,7,8],:,0]
    # f[4,:,-2] = f[2,:,-1]
    # f[7,:,-2] = f[5,:,-1]
    # f[8,:,-2] = f[6,:,-1]

    return f

def Pressure(g, c, w, pin, pout):
    #Pressure In
    rhoN = np.einsum('ij->j',g[:,-2,:])
    uN = np.einsum('ia,iy->ay',c,g[:,-2,:])/rhoN
    cdot3u = 3*np.einsum('ia,ay->iy',c,uN)
    usq = np.einsum('ay->y',uN*uN)
    feqpin = pin*w[:,np.newaxis]*(1+cdot3u*(1+0.5*cdot3u)-1.5*usq[np.newaxis,:])
    wrhoN = np.einsum('i,y->iy',w,rhoN)
    feqN = wrhoN*(1+cdot3u*(1+0.5*cdot3u)-1.5*usq[np.newaxis,:])
    fneqN = g[:,-2,:]
    fin = feqpin+(fneqN-feqN)
    #
    #Pressure Out
    rho1 = np.einsum('ij->j',g[:,1,:])
    u1 = np.einsum('ia,iy->ay',c,g[:,1,:])/rho1
    cdot3u = 3*np.einsum('ia,ay->iy',c,u1)
    usq = np.einsum('ay->y',u1*u1)
    feqpout = pout*w[:,np.newaxis]*(1+cdot3u*(1+0.5*cdot3u)-1.5*usq[np.newaxis,:])
    wrho1 = np.einsum('i,y->iy',w,rho1)
    feq1 = wrho1*(1+cdot3u*(1+0.5*cdot3u)-1.5*usq[np.newaxis,:])
    fneq1 = g[:,1,:]
    fout = feqpout+(fneq1-feq1)
    return fin, fout

### Calculation
starttime = time.time()
fig = plt.figure()
ax = fig.add_subplot()
uy_anal = []
f = equilibrium(rho, u)
for i in range(timestep):
    f = streaming(f, c)
    f = bounce_back(f)
    rho, uy_kl, ux_kl = collision(f, omega)
    f[:,0,:], f[:,-1,:] = Pressure(f, c, weights, pin, pout)

    if i%100==0: #every 20 timesteps
        rho = np.einsum('ijk->jk', f)
        u = np.einsum('ia,ixy->axy',c,f)/rho
        ax.set_xlabel('Position in the cross section')
        ax.set_ylabel('Velocity')
        ax.plot(u[0,5,1:-1], label = str()) #ax.plot(u[0,5,1:-2])

    # if i == np.arange(timestep)[-1]:
    #     for j in range(len(y)):
    #         #uy_anal.append(((1*epsilon)/(3*rho[:,j+1]*nu*Nx)) * (y[j]*(Ny-y[j]))) #analytical velocity profile
    #     ax.plot(uy_anal, label='Analytical', color = 'blue', linestyle='--')
    #     #ax.legend()

uy_anal = (epsilon/Nx/nu) * y*(Ny-y)/3.
ax.plot(uy_anal, label='Analytical', color = 'blue', linestyle='--')

endtime = time.time() 
plt.xlim([0,30])
plt.savefig('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images/MS5 Poiseuille Flow Velocity' + str(i).zfill(4))
print('{} timesteps took {}s' .format(i,endtime-starttime)) #Calculate how long it take to run the main loop