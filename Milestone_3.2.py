"""
HPC Milestone 3
Jared Engelken
Python 3
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

dtype = np.float32 #data type
timestep = 100 #number of animation iterations
Nx = 30 #plot size in the x direction #15
Ny = 30 #plot size in the y direction #10
X,Y = np.meshgrid(range(Ny),range(Nx)) #create the plot grid

c = np.array([[0,  1,  0, -1,  0,  1, -1, -1,  1],    # velocities, x components
              [0,  0,  1,  0, -1,  1,  1, -1, -1]]).T # velocities, y components
weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) # weights for each channel
omega = 0.3  #relaxation rate #0.3 #A higher omega means a lower kinematic viscosity. Anything above 1.7 becomes unstable!
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
amplitude = []
rhoAnim = np.zeros((Nx, Ny, timestep))
uAnim = np.zeros((Nx, Ny, timestep))
for i in range(timestep):
    streaming(f)
    rho, ux_kl, uy_kl = collide(f, omega)
    rhoAnim[:,:,i] = rho
    uAnim[:,:,i] = uy_kl
    # Fourier analysis
    amplitude += [(uy_kl[:, Ny//2]*uy_k).sum() * 2/Nx]
    #if i%10 == 0:
        #fig = plt.figure(figsize=(Nx,Ny))
        #ax = fig.add_subplot(111, projection='3d')
        #ax.set_zlim3d(0.995,1.005)
        #ax.plot_wireframe(X,Y,rho) 
        #ax.set_zlim3d(-1,1)
        #ax.plot_surface(X,Y,uy_kl)    
        #plt.show()
        #fig.savefig('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images'
        #+'\Figure '+str(i)+'.png', bbox_inches='tight')   # save the figure to file

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
    +'\Density_Animation.gif',writer='imagemagick')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
plot1 = [ax.plot_surface(X, Y, uAnim[:,:,0], color='0.75', rstride=1, cstride=1)]
ax1.set_zlim3d(-1,1)
ax1.set_xlabel('X Direction')
ax1.set_ylabel('Y Direction')
ax1.set_title('Shear Wave Velocity Decay')
anim1 = animation.FuncAnimation(fig1, animate1, timestep, fargs=(uAnim, plot))
fn1 = 'plot_surface_animation_funcanimation'
anim1.save('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images'
    +'\Velocity_Animation.gif',writer='imagemagick')

# print(amplitude)
fig2 = plt.figure(figsize=(8,8))
fig2.subplots_adjust(top=0.8)
ax2 = fig2.add_subplot(211)
ax2.set_xlabel('Time')
ax2.set_ylabel('Amplitude')
ax2.set_title('Shear Wave Amplitude Decay')
ax2.plot(range(timestep),amplitude)
plt.savefig('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images'
    +'\Amplitude.png',writer='imagemagick',bbox_inches='tight')
# plt.show()
