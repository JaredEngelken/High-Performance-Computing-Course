"""
HPC Milestone 3
Jared Engelken
Python 3
"""
### Ignore any gifs that are produced except for the first omega value!

import numpy as np
import matplotlib.pyplot as plt

dtype = np.float32 #data type
timestep = 500 #number of iterations
Nx = 15 #plot size in the x direction
Ny = 10 #plot size in the y direction
X,Y = np.meshgrid(range(Ny),range(Nx)) #create the plot grid

c = np.array([[0,  1,  0, -1,  0,  1, -1, -1,  1],    # velocities, x components
              [0,  0,  1,  0, -1,  1,  1, -1, -1]]).T # velocities, y components
weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) # weights for each channel
omega = [0.5,1.0,1.5]  #relaxation rate - A higher omega means a lower kinematic viscosity. Anything above 1.7 becomes unstable!
epsilon = 0.001 #pressure difference
rho0 = 1.0 #rest density

### Compute functions
def equilibrium(rho_eq, ux_eq, uy_eq): #equilibrium approximation
    cu5 = ux_eq + uy_eq #velocity for channel 5
    cu6 = -ux_eq + uy_eq #velocity for channel 6
    cu7 = -ux_eq - uy_eq #velocity for channel 7
    cu8 = ux_eq - uy_eq #velocity for channel 8
    uu_eq = ux_eq**2 + uy_eq**2 #sum of the x and y squared velocities
    #returns feq as an array using the discretized equilibrium function
    return np.array([weights[0]*rho_eq*(1 - 3/2*uu_eq), 
                     weights[1]*rho_eq*(1 + 3*ux_eq + 9/2*ux_eq**2 - 3/2*uu_eq),
                     weights[2]*rho_eq*(1 + 3*uy_eq + 9/2*uy_eq**2 - 3/2*uu_eq),
                     weights[3]*rho_eq*(1 - 3*ux_eq + 9/2*ux_eq**2 - 3/2*uu_eq),
                     weights[4]*rho_eq*(1 - 3*uy_eq + 9/2*uy_eq**2 - 3/2*uu_eq),
                     weights[5]*rho_eq*(1 + 3*cu5 + 9/2*cu5**2 - 3/2*uu_eq),
                     weights[6]*rho_eq*(1 + 3*cu6 + 9/2*cu6**2 - 3/2*uu_eq),
                     weights[7]*rho_eq*(1 + 3*cu7 + 9/2*cu7**2 - 3/2*uu_eq),
                     weights[8]*rho_eq*(1 + 3*cu8 + 9/2*cu8**2 - 3/2*uu_eq)])

def collision(f, omega): #collision 
    rho_coll = np.sum(f, axis=0) #recalculate the local density
    ux_coll = (f[1]-f[3]+f[5]-f[6]-f[7]+f[8])/rho_coll #calculate the local average velocity for x
    uy_coll = (f[2]-f[4]+f[5]+f[6]-f[7]-f[8])/rho_coll #calculate the local average velocity for y
    f += omega*(equilibrium(rho_coll, ux_coll, uy_coll)-f) #recalculate f with the discretized BTE
    return rho_coll, ux_coll, uy_coll

def streaming(f): #streaming
    for i in range(1, 9): #for each of the 9 velocity channels
        f[i] = np.roll(f[i], c[i], axis=(0, 1)) #transport the particles on grid f by 1 lattice point 

### Main Loop & Plots
v1 = []
amp = []
ampl = []
amplitude = []
vis = []
viscosity = []
rhoTop3 = np.zeros((3,timestep))
rhoTop = []
uTop3 = np.zeros((3,timestep))
uTop = []
for om in omega:
    ### Initialize probability distribution with a shear wave
    x_k = np.arange(Nx)
    wavevector = 2*np.pi/Nx
    uy_k = np.sin(wavevector*x_k, dtype=dtype)
    rho = np.ones((Nx,Ny))
    for j in range(Ny):
        rho[:,j]= rho0 + epsilon * np.sin(wavevector*x_k, dtype=dtype)

    ux = np.zeros_like(uy_k).reshape((Nx, 1))
    uy = uy_k.reshape((Nx, 1))
    
    f = equilibrium(rho, ux, uy)

    for i in range(timestep):
        streaming(f)
        rho, ux_kl, uy_kl = collision(f, om)
        rhoTop.append(rho[4,0])
        uTop.append(uy_kl[4,0])
        # Fourier analysis
        amp += [(uy_kl[:, Ny//2]*uy_k).sum() * 2/Nx]
        ampl = amp/amp[0]
        vis = (np.log(ampl)/((2*np.pi/Nx)**2*i))
    
    v1.append(np.full(timestep,(1/3)*(1/om-.5))) #analytical predicition of shear viscosity
    amplitude.append(ampl)
    viscosity.append(vis)
    step = int(om*2-1)
    rhoTop3[step] = rhoTop
    uTop3[step] = uTop
    
    rhoTop.clear()
    uTop.clear()
    amp = []
    ampl = []
    vis = []

### Graphs
fig2 = plt.figure(figsize=(8,8))
fig2.subplots_adjust(top=0.8)
ax2 = fig2.add_subplot(211)
ax2.set_xlabel('Time')
ax2.set_ylabel('Viscocity (Amplitude)')
ax2.set_title('Shear Wave Viscosity Amplitude Decay')
ax2.plot(range(timestep),v1[0], label='Analyitcal Viscosity Prediction | omega='+str(omega[0]), color = 'pink')
ax2.plot(range(timestep),amplitude[0], label='Amplitude | omega='+str(omega[0]), color = 'red')
ax2.plot(range(timestep),v1[1], label='Analyitcal Viscosity Prediction | omega='+str(omega[1]), color = 'lightgreen')
ax2.plot(range(timestep),amplitude[1], label='Amplitude | omega='+str(omega[1]), color = 'green')
ax2.plot(range(timestep),v1[2], label='Analyitcal Viscosity Prediction | omega='+str(omega[2]), color = 'lightblue')
ax2.plot(range(timestep),amplitude[2], label='Amplitude | omega='+str(omega[2]), color = 'blue')
ax2.legend(loc='upper right', prop={'size': 6})
plt.savefig('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images'
    +'/Amplitude.png',writer='imagemagick',bbox_inches='tight')

fig3 = plt.figure()
ax3 = fig3.add_subplot()
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Density (Amplitude)\nρ0={}, Ɛ={}'.format(rho0,epsilon))
ax3.set_title('Shear Wave Density Decay')
ax3.plot(range(timestep),rhoTop3[2], label='Amplitude | ω='+str(omega[2]), color = 'lightblue')
ax3.plot(range(timestep),rhoTop3[1], label='Amplitude | ω='+str(omega[1]), color = 'green')
ax3.plot(range(timestep),rhoTop3[0], label='Amplitude | ω='+str(omega[0]), color = 'red')
ax3.legend(loc='upper right', prop={'size': 6})
plt.savefig('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images'
    +'\MS3_density_decay.png',writer='imagemagick',bbox_inches='tight')

fig4 = plt.figure()
ax4 = fig4.add_subplot()
ax4.set_xlabel('Time Step')
ax4.set_ylabel('Velocity (Amplitude)\nρ0={}, Ɛ={}'.format(rho0,epsilon))
ax4.set_title('Shear Wave Velocity Decay')
ax4.plot(range(timestep),uTop3[2], label='Amplitude | ω='+str(omega[2]), color = 'lightblue')
ax4.plot(range(timestep),uTop3[1], label='Amplitude | ω='+str(omega[1]), color = 'green')
ax4.plot(range(timestep),uTop3[0], label='Amplitude | ω='+str(omega[0]), color = 'red')
ax4.legend(loc='upper right', prop={'size': 6})
plt.savefig('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images'
    +'\MS3_velocity_decay.png',writer='imagemagick',bbox_inches='tight')