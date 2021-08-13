"""
HPC Milestone 4
Jared Engelken
Python 3
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


### Set Parameters, define grid & set initialization values
dtype = np.float32 #data type
timestep = 500 #number of animation iterations
Nx = 30 #plot size in the x direction #15
Ny = 30 #plot size in the y direction #10
Y,X = np.meshgrid(range(Ny),range(Nx)) #create the plot grid

c = np.array([[0,  1,  0, -1,  0,  1, -1, -1,  1],    # velocities, x components
              [0,  0,  1,  0, -1,  1,  1, -1, -1]]).T # velocities, y components
weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) # weights for each channel
omega = 0.5  #relaxation rate #0.3 #A higher omega means a lower kinematic viscosity. Anything above 1.7 becomes unstable!
epsilon = 0.001 #pressure difference
wall_vel = 0.1 #wall speed

# Define starting functions & values for u & rho
x_k = np.arange(Nx)
wavevector = 2*np.pi/Nx
uy_k = np.sin(wavevector*x_k, dtype=dtype)

rho = np.ones((Nx,Ny))    

ux = np.zeros_like(uy_k).reshape((Nx, 1))
uy = np.zeros_like(uy_k).reshape((Nx, 1))

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

def bounce_back (f, wall_vel): #boundary conditions and particle bounce back
    #before streaming the particles out of bounds, make a copy of their current position
    f_bottom = f[:,:,0].copy() #copy the current bottom index of the (N,M,9) f array
    f_top = f[:,:,-1].copy() #copy the current top index of the (N,M,9) f array
    f_left = f[:,0,:].copy() #copy the current left index of the (N,M,9) f array
    f_right = f[:,-1,:].copy() #copy the current right index of the (N,M,9) f array

    streaming(f) #stream the particles
    #for the bottom wall, all particles that would stream out, reflect in the inverse direction
    f[[2,5,6],:,0] = f_bottom[[4,7,8],:] 
    
    #recalculate the density of particles along the top boundary after the bounce back
    #sum of upwards moving particles before streaming and all particles not moving downwards after streaming
    rho_wall = f_top[2]+f_top[5]+f_top[6]+f[0,:,-1]+f[1,:,-1]+f[2,:,-1]+f[3,:,-1]+f[5,:,-1]+f[6,:,-1]
    f[4,:,-1] = f_top[4] #particles colliding with the lid directly then bounce back directly
    #particles colliding with the wall at an angle bounce back with an altered velocity
    f[7,:,-1] = f_top[5] - 6*weights[7]*rho_wall*wall_vel #the lid slows down particles moving left
    f[8,:,-1] = f_top[6] + 6*weights[8]*rho_wall*wall_vel #the lid speeds up particles moving right
    
    #the rest of the walls - change to "if False" for Couette flow test, otherwise set to "if True"
    if False:
        #left wall
        f[1,0,:] = f_left[3] #particles streaming west bounce back east
        f[5,0,:] = f_left[7] #particles streaming southwest bounce back northeast
        f[8,0,:] = f_left[6] #particles streaming northwest bounce back southeast

        #right wall
        f[3,-1,:] = f_right[1] #particles streaming east bounce back west
        f[6,-1,:] = f_right[8] #particles streaming southeast bounce back northwest
        f[7,-1,:] = f_right[5] #particles streaming northeast bounce back southwest

        #bottom-left corner
        f[2,0,0] = f_bottom[4,0] #particles streaming south bounce back north
        f[1,0,0] = f_bottom[3,0] #particles streaming west bounce back east
        f[5,0,0] = f_bottom[7,0] #particles streaming southwest bounce back northeast

        #bottom-right corner
        f[2,-1,0] = f_bottom[4,-1] #particles streaming south bounce back north
        f[3,-1,0] = f_bottom[1,-1] #particles streaming west bounce back east
        f[6,-1,0] = f_bottom[8,-1] #particles streaming southeast bounce back northwest

        #top-left corner
        f[4,0,-1] = f_top[2,0] #particles streaming north bounce back south
        f[1,0,-1] = f_top[3,0] #particles streaming west bounce back east
        f[8,0,-1] = f_top[6,0] #particles streaming northwest bounce back southeast

        #top-right corner
        f[4,-1,-1] = f_top[2,-1] #particles streaming north bounce back south
        f[3,-1,-1] = f_top[1,-1] #particles streaming east bounce back west
        f[7,-1,-1] = f_top[5,-1] #particles streaming northeast bounce back southwest
     

def animate(frame_number, uxAnim, plot):
    plot[0].remove()
    plot[0] = ax2.plot_surface(X, Y, uxAnim[:,:,frame_number], cmap="plasma")

### Calculation
f = equilibrium(rho, ux, uy)


### Main Loop & Plots
uxAnim = np.zeros((Nx, Ny, timestep))
uxPlot = np.zeros((Nx, Ny, timestep))
for i in range(timestep):
    bounce_back(f, wall_vel)
    rho_kl, ux_kl, uy_kl = collision(f, omega)
    uxAnim[:,:,i] = ux_kl

    if i%20==0:
        uxPlot[:,:,i] = ux_kl

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.text(Nx/2, Ny+1, 'Moving Wall', c = 'red', horizontalalignment = 'center')
        ax.text(Nx + 1, Ny/2, 'Outlet', c = 'blue',  verticalalignment='center')
        ax.text(0-6, Ny/2, 'Inlet', c = 'blue',  verticalalignment='center')
        ax.text(Nx/2, 0 - 5, 'Bottom Wall', c = 'blue', horizontalalignment = 'center')
        ax.quiver(X,Y,ux_kl,uy_kl)
        plt.savefig('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images/MS4 Density Over Time' + str(i).zfill(4))

        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(111,projection = '3d')
        # # ax1.set_zlim([-0.1,0.1])
        # ax1.set_xlabel('x-Axis (Point in Grid')
        # ax1.set_ylabel('y-Axis (Point in Grid)')
        # ax1.set_zlabel('Velocity')
        # ax1.plot_surface(X,Y,ux_kl)
        # plt.savefig('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images/Figures_2_Velocity' + str(i).zfill(4))

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
plot = [ax2.plot_surface(X, Y, uxAnim[:,:,0], color='0.75', rstride=1, cstride=1)]
ax2.set_xlabel('X Direction')
ax2.set_ylabel('Y Direction')
ax2.set_zlabel('Velocity')
ax2.set_title('Velocity Profile Over Time')
anim = animation.FuncAnimation(fig2, animate, timestep, fargs=(uxAnim, plot))
anim.save('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images'
    +'\MS4 Velocity Animation.gif',writer='imagemagick')

fig3 = plt.figure()
ax3 = fig3.add_subplot()
ax3.set_xlabel('X Direction')
ax3.set_ylabel('Velocity')
ax3.set_title('Velocity Profile Over Time')
for j in range(timestep):
    ax3.plot(range(Ny),uxPlot[1,:,j])
plt.savefig('C:\MSc Sustainable Materials - Polymers\High Performance Computing\Images'
    +'\MS4 Velocity Profile Over Time.png',writer='imagemagick',bbox_inches='tight')
