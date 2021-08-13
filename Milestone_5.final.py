"""
HPC Milestone 5
Jared Engelken
Python 3
"""
import numpy as np
import matplotlib.pyplot as plt
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
rho = rho0*np.ones((Nx+2,Ny+2))
u = np.zeros((2,Nx+2,Ny+2))

def equilibrium(rho_eq, u_eq): #equilibrium approximation
    cu_eq= np.dot(u_eq.T, c.T).T #dot product of channel velocities
    uu_eq= np.sum(u_eq**2, axis=0) #sum of square velocities
    return(weights*(rho_eq*(1 + 3*cu_eq+ 9/2*cu_eq**2 - 3/2*uu_eq)).T).T #calculate the fluid particle equilibrium distribution

def collision(f, omega): #collision 
    rho_coll = np.sum(f, axis=0) #recalculate the local density
    ux_coll = (f[1]-f[3]+f[5]-f[6]-f[7]+f[8])/rho_coll #calculate the local average velocity for x
    uy_coll = (f[2]-f[4]+f[5]+f[6]-f[7]-f[8])/rho_coll #calculate the local average velocity for y
    f += omega*(equilibrium(rho_coll, ux_coll, uy_coll)-f) #recalculate f with the discretized BTE
    return rho_coll, ux_coll, uy_coll

def streaming(f): #streaming
    for i in range(1, 9): #for each of the 9 velocity channels
        f[i] = np.roll(f[i], c[i], axis=(0, 1)) #transport the particles on grid f by 1 lattice point 

def bounce_back (f):
    ### This code works
    f_bottom = f[:,:,0].copy() #copy the current bottom index of the (N,M,9) f array
    f_top = f[:,:,-1].copy() #copy the current top index of the (N,M,9) f array

    # Defining the boundary conditions --> Anti Direction for hitting the top and bottom wall
    f[[2,5,6],:,1] = f_bottom[[4,7,8],:] #for the bottom wall, all particles that would stream out, reflect in the inverse direction
    f[4,:,-2] = f_top[2,:] #particles colliding with the top directly then bounce back directly
    f[8,1:-1,-2] = f_top[6,:-2] #particles streaming northeast bounce back southwest
    f[7,1:-1,-2] = f_top[5,:-2] #particles streaming northwest bounce back southeast

    ### This code below also works
    # f[[2,5,6],:,1] = f[[4,7,8],:,0]
    # f[4,:,-2] = f[2,:,-1]
    # f[7,:,-2] = f[5,:,-1]
    # f[8,:,-2] = f[6,:,-1]

    return f

def Pressure(g, c, w, pin, pout): #create the pressure difference in the 'pipe'
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

### Main Loop & Plots
starttime = time.time()
fig = plt.figure()
ax = fig.add_subplot()
uy_anal = []
f = equilibrium(rho, u)
for i in range(timestep):
    streaming(f)
    f = bounce_back(f)
    rho, uy_kl, ux_kl = collision(f, omega)
    f[:,0,:], f[:,-1,:] = Pressure(f, c, weights, pin, pout)

    if i%100==0:
        rho = np.einsum('ijk->jk', f)
        u = np.einsum('ia,ixy->axy',c,f)/rho
        ax.set_xlabel('Position in the cross section')
        ax.set_ylabel('Velocity')
        ax.plot(u[0,5,1:-1], label = str())

uy_anal = (epsilon/Nx/nu) * y*(Ny-y)/3.
ax.plot(uy_anal, label='Analytical', color = 'blue', linestyle='--')

endtime = time.time() 
plt.xlim([0,30])
plt.savefig('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images/MS5 Poiseuille Flow Velocity' + str(i).zfill(4))
print('{} timesteps took {}s' .format(i,endtime-starttime)) #Calculate how long it take to run the main loop