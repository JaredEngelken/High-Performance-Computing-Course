"""
HPC Milestone 3
Jared Engelken
Python 3
"""
### Ignore any gifs that are produced except for the first omega value!

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

dtype = np.float32 #data type
timestep = 100 #number of animation iterations
Nx = 15 #plot size in the x direction #15
Ny = 10 #plot size in the y direction #10
X,Y = np.meshgrid(range(Ny),range(Nx)) #create the plot grid

c = np.array([[0,  1,  0, -1,  0,  1, -1, -1,  1],    # velocities, x components
              [0,  0,  1,  0, -1,  1,  1, -1, -1]]).T # velocities, y components
weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) # weights for each channel
omega = [0.5,1.0,1.5]  #relaxation rate #0.3 #A higher omega means a lower kinematic viscosity. Anything above 1.7 becomes unstable!
epsilon = 0.01 #pressure difference
rho0 = 1.0 #rest density

### Compute functions
def equilibrium(rho, ux, uy):
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

def collide(f, omega):
    rho = np.sum(f, axis=0)
    ux = (f[1] - f[3] + f[5] - f[6] - f[7] + f[8])/rho
    uy = (f[2] - f[4] + f[5] + f[6] - f[7] - f[8])/rho
    f += omega*(equilibrium(rho, ux, uy) - f)
    return rho, ux, uy

def streaming(f):
    for i in range(1, 9):
        f[i] = np.roll(f[i], c[i], axis=(0, 1))

### Initialize probability distribution with a shear wave
x_k = np.arange(Nx)
wavevector = 2*np.pi/Nx
uy_k = np.sin(wavevector*x_k, dtype=dtype)

rho = np.ones((Nx,Ny))
# rho = np.sum(f,axis=2) #calculate the density at each lattice point
for j in range(Ny):
    rho[:,j]= rho0 + epsilon * np.sin(wavevector*x_k, dtype=dtype)

ux = np.zeros_like(uy_k).reshape((Nx, 1))
uy = uy_k.reshape((Nx, 1))

f = equilibrium(rho, ux, uy)

def animate(frame_number, rhoAnim, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(X, Y, rhoAnim[:,:,frame_number], cmap="plasma")

def animate1(frame_number, uAnim, plot):
    plot[0].remove()
    plot[0] = ax1.plot_surface(X, Y, uAnim[:,:,frame_number], cmap="plasma")

### Main loop
v1 = []
amp = []
ampl = []
amplitude = []
vis = []
viscosity = []
rhoAnim = np.zeros((Nx, Ny, timestep))
uAnim = np.zeros((Nx, Ny, timestep))
for om in omega:
    for i in range(timestep):
        streaming(f)
        rho, ux_kl, uy_kl = collide(f, om)
        rhoAnim[:,:,i] = rho
        uAnim[:,:,i] = uy_kl
        # Fourier analysis
        amp += [(uy_kl[:, Ny//2]*uy_k).sum() * 2/Nx]
        ampl = amp/amp[0]
        vis = (np.log(ampl)/((2*np.pi/Nx)**2*i))

        ### Frame-by-Frame
        ### Comment this section out if printing the gifs
        # if i%10 == 0:
        #     fig = plt.figure(figsize=(Nx,Ny))
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.set_zlim3d(0.995,1.005)
        #     ax.set_xlabel('X Direction', fontsize=30)
        #     ax.set_ylabel('Y Direction', fontsize=30)
        #     ax.set_title('Shear Wave Density Decay', fontsize=40)
        #     ax.plot_wireframe(X,Y,rho)    
        #     # plt.show()
        #     fig.savefig('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images'
        #     +'\Density Decay '+str(i)+'.png', bbox_inches='tight')   # save the figure to file

        #     fig1 = plt.figure(figsize=(Nx,Ny))
        #     axi = fig1.add_subplot(111, projection='3d')
        #     axi.set_zlim3d(-1,1)
        #     axi.set_xlabel('X Direction', fontsize=30)
        #     axi.set_ylabel('Y Direction', fontsize=30)
        #     axi.set_title('Shear Wave Velocity Decay', fontsize=40)
        #     axi.plot_wireframe(X,Y,uy_kl)    
        #     # plt.show()
        #     fig1.savefig('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images'
        #     +'\Velocity Decay '+str(i)+'.png', bbox_inches='tight')   # save the figure to file

    
    
    v1.append(np.full(timestep,(1/3)*(1/om-.5)))
    amplitude.append(ampl)
    viscosity.append(vis)

    ### Gifs
    ### Comment this section out if printing frame-by-frame
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot = [ax.plot_surface(X, Y, rhoAnim[:,:,0], color='0.75', rstride=1, cstride=1)]
    ax.set_zlim3d(0.995,1.005)
    ax.set_xlabel('X Direction')
    ax.set_ylabel('Y Direction')
    ax.set_title('Shear Wave Density Decay')
    anim = animation.FuncAnimation(fig, animate, timestep, fargs=(rhoAnim, plot))
    fn = 'plot_surface_animation_funcanimation'
    anim.save('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images'
        +'\Density_Animation '+str(om)+'.gif',writer='imagemagick')

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    plot1 = [ax.plot_surface(X, Y, uAnim[:,:,0], color='0.75', rstride=1, cstride=1)]
    ax1.set_zlim3d(-1,1)
    ax1.set_xlabel('X Direction')
    ax1.set_ylabel('Y Direction')
    ax1.set_title('Shear Wave Velocity Decay')
    anim1 = animation.FuncAnimation(fig1, animate1, timestep, fargs=(uAnim, plot1))
    fn1 = 'plot_surface_animation_funcanimation'
    anim1.save('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images'
        +'\Velocity_Animation '+str(om)+'.gif',writer='imagemagick')

    amp = []
    ampl = []
    vis = []
    rhoAnim = np.zeros((Nx, Ny, timestep))
    uAnim = np.zeros((Nx, Ny, timestep))


### Viscosity Amplitude Decay Graph
fig2 = plt.figure(figsize=(8,8))
fig2.subplots_adjust(top=0.8)
ax2 = fig2.add_subplot(211)
ax2.set_xlabel('Time')
ax2.set_ylabel('Viscocity (Amplitude)')
ax2.set_title('Shear Wave Viscosity Amplitude Decay')
ax2.plot(range(timestep),v1[0], label='Analyitcal Viscosity Prediction | omega='+str(omega[0]), color = 'pink')
ax2.plot(range(timestep),amplitude[0], label='Amplitude | omega='+str(omega[0]), color = 'red')
# ax2.plot(range(timestep),viscosity[0], label='Viscosity | omega='+str(omega[0]), color = 'darkred')
ax2.plot(range(timestep),v1[1], label='Analyitcal Viscosity Prediction | omega='+str(omega[1]), color = 'lightgreen')
ax2.plot(range(timestep),amplitude[1], label='Amplitude | omega='+str(omega[1]), color = 'green')
# ax2.plot(range(timestep),viscosity[1], label='Viscosity | omega='+str(omega[1]), color = 'darkgreen')
ax2.plot(range(timestep),v1[2], label='Analyitcal Viscosity Prediction | omega='+str(omega[2]), color = 'lightblue')
ax2.plot(range(timestep),amplitude[2], label='Amplitude | omega='+str(omega[2]), color = 'blue')
# ax2.plot(range(timestep),viscosity[2], label='Viscosity | omega='+str(omega[2]), color = 'darkblue')
ax2.legend(loc='upper right', prop={'size': 6})
plt.savefig('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images'
    +'\Amplitude.png',writer='imagemagick',bbox_inches='tight')