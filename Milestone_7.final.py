"""
HPC Milestone 7
Jared Engelken
Python 3
"""
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import time

starttime = time.time() #start program timer

### Set Parameters, define grid & set initialization values
dtype = np.float32 #data type
timestep = 10000 #number of animation iterations
Nx = 300 #plot size in the x direction
Ny = 300 #plot size in the y direction
Y,X = np.meshgrid(range(Ny),range(Nx)) #create the plot grid
c = np.array([[0,  1,  0, -1,  0,  1, -1, -1,  1],    # velocity channels, x components
              [0,  0,  1,  0, -1,  1,  1, -1, -1]]).T # velocity channels, y components
weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) # weights for each channel
omega = 1.0  #relaxation rate - A higher omega means a lower kinematic viscosity. Anything above 1.7 becomes unstable!
epsilon = 0.001 #pressure difference
wall_vel = 0.1 #wall speed

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
    if True:
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

def communicate(f_com):
    #communicate boundary indices to ghost indices
    #send left
    ghost = f_com[:, -1, :].copy() #create a ghost index on the right to receive incoming data
    comm.Sendrecv(f_com[:, 1, :].copy(), left_dest, #send leftmost index to left_dest
                  recvbuf=ghost, source=left_source) #receive leftmost index from left_source
    f_com[:, -1, :] = ghost #set the rightmost index to the received data
    #send right
    ghost = f_com[:, 0, :].copy() #create a ghost index on the left to receive incoming data
    comm.Sendrecv(f_com[:, -2, :].copy(), right_dest, #send rightmost index to right_dest
                  recvbuf=ghost, source=right_source) #receive rightmost index from right_source
    f_com[:, 0, :] = ghost #set the leftmost index to the received data
    #send bottom
    ghost = f_com[:, :, -1].copy() #create a ghost index on the top to receive incoming data
    comm.Sendrecv(f_com[:, :, 1].copy(), bottom_dest, #send top index to bottom_dest
                  recvbuf=ghost, source=bottom_source) #receive top index from bottom_source
    f_com[:, :, -1] = ghost #set the top index to the received data
    #send top
    ghost = f_com[:, :, 0].copy() #create a ghost index on the bottom to receive incoming data
    comm.Sendrecv(f_com[:, :, -2].copy(), top_dest, #send bottom index to top_dest
                  recvbuf=ghost, source=top_source) #receive bottom index from top_source
    f_com[:, :, 0] = ghost #set the bottom index to the received data

def save_mpiio(comm, fn, g_kl):
    """
    Write a global two-dimensional array to a single file in the npy format
    using MPI I/O: https://docs.scipy.org/doc/numpy/neps/npy-format.html

    Arrays written with this function can be read with numpy.load.

    Parameters
    ----------
    comm
        MPI communicator.
    fn : str
        File name.
    g_kl : array_like
        Portion of the array on this MPI processes. This needs to be a
        two-dimensional array.
    """
    from numpy.lib.format import dtype_to_descr, magic
    magic_str = magic(1, 0)

    local_nx, local_ny = g_kl.shape
    nx = np.empty_like(local_nx)
    ny = np.empty_like(local_ny)

    commx = comm.Sub((True, False))
    commy = comm.Sub((False, True))
    commx.Allreduce(np.asarray(local_nx), nx)
    commy.Allreduce(np.asarray(local_ny), ny)

    arr_dict_str = str({ 'descr': dtype_to_descr(g_kl.dtype),
                         'fortran_order': False,
                         'shape': (np.asscalar(nx), np.asscalar(ny)) })
    while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
        arr_dict_str += ' '
    arr_dict_str += '\n'
    header_len = len(arr_dict_str) + len(magic_str) + 2

    offsetx = np.zeros_like(local_nx)
    commx.Exscan(np.asarray(ny*local_nx), offsetx)
    offsety = np.zeros_like(local_ny)
    commy.Exscan(np.asarray(local_ny), offsety)

    file = MPI.File.Open(comm, fn, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    if comm.Get_rank() == 0:
        file.Write(magic_str)
        file.Write(np.int16(len(arr_dict_str)))
        file.Write(arr_dict_str.encode('latin-1'))
    mpitype = MPI._typedict[g_kl.dtype.char]
    filetype = mpitype.Create_vector(g_kl.shape[0], g_kl.shape[1], ny)
    filetype.Commit()
    file.Set_view(header_len + (offsety+offsetx)*mpitype.Get_size(),
                  filetype=filetype)
    file.Write_all(g_kl.copy())
    filetype.Free()
    file.Close()


### Parallelization
size = MPI.COMM_WORLD.Get_size() #get the number of MPI processes
rank = MPI.COMM_WORLD.Get_rank() #get the ranks of the MPI processes
print('Size: ', size)
print('rank: ', rank)

ndx = int(np.sqrt(size)) #set the number of x partitions to sqrt(size)
ndy = int(np.sqrt(size)) #set the number of y partitions to sqrt(size)
print('Ndx = ', ndx)
print('Ndy = ', ndy)

if rank == 0:
    print('Running in parallel on {} MPI processes.'.format(size))
assert ndx*ndy == size #make sure that the number of partitions is the number of processes

#create cartesian communicator and get MPI ranks of neighboring cells
comm = MPI.COMM_WORLD.Create_cart((ndx, ndy), periods=(False, False))
print('rank = {}/{}'.format(comm.Get_rank(), comm.Get_size()))

local_nx = Nx//ndx #split the x axis of the lattice into ndx partitions
local_ny = Ny//ndy #split the y axis of the lattice into ndy partitions
print("local Nx = ", local_nx)
print("local Ny = ", local_ny)

ux = np.zeros(local_nx) #initialize the local velocities for the x direction
ux.shape = (local_nx,1)
print('ux shape: ',ux.shape)

uy = np.zeros(local_ny) #initialize the local velocities for the y direction
uy.shape = (local_ny,1)
print('uy shape: ', uy.shape)

rho=np.ones((local_nx, local_ny)) #initialize the local density
print('rho: ', rho.shape)

#assign the communication directions
left_source, left_dest = comm.Shift(0, -1)
right_source, right_dest = comm.Shift(0, 1)
bottom_source, bottom_dest = comm.Shift(1, -1)
top_source, top_dest = comm.Shift(1, 1)

# We need to take care that the total number of *local* grid points sums up to
# nx. The right and topmost MPI processes are adjusted such that this is
# fulfilled even if nx, ny is not divisible by the number of MPI processes.
if right_dest < 0:
    # This is the rightmost MPI process
    local_nx = Nx - local_nx*(ndx-1)
without_ghosts_x = slice(0, local_nx)
if right_dest >= 0:
    # Add ghost cell
    local_nx += 1
if left_dest >= 0:
    # Add ghost cell
    local_nx += 1
    without_ghosts_x = slice(1, local_nx+1)
if top_dest < 0:
    # This is the topmost MPI process
    local_ny = Ny - local_ny*(ndy-1)
without_ghosts_y = slice(0, local_ny)
if top_dest >= 0:
    # Add ghost cell
    local_ny += 1
if bottom_dest >= 0:
    # Add ghost cell
    local_ny += 1
    without_ghosts_y = slice(1, local_ny+1)

mpix, mpiy = comm.Get_coords(rank)
print('Rank {} has domain coordinates {}x{} and a local grid of size {}x{} (including ghost cells).'.format(rank, mpix, mpiy, local_nx, local_ny))

gridpoints = Nx*Ny
print('Number of grid points: ', gridpoints)
print('Time Steps: ', timestep)

### Initial Calculation
f = equilibrium(rho, ux, uy)

### Main Loop & Plots
for i in range(timestep):
    communicate(f)
    bounce_back(f, wall_vel)
    rho_calc, ux_calc, uy_calc = collision(f, omega)

    if (i % 100 == 0 and i != 0) or i == timestep-1: #save a plot every 100 time steps
        save_mpiio(comm, 'ux_{}.npy'.format(i), ux_calc[without_ghosts_x, without_ghosts_y])
        save_mpiio(comm, 'uy_{}.npy'.format(i), uy_calc[without_ghosts_x, without_ghosts_y])
      
        ux_calc = np.load('ux_{}.npy'.format(i))
        uy_calc = np.load('uy_{}.npy'.format(i))
        
        nx, ny = ux_calc.shape
        
        plt.figure()
        xx = np.arange(nx)
        yy = np.arange(ny)
        plt.xlim((0,Nx))
        plt.ylim((0,Ny))
        plt.suptitle('Lattice Boltzmann Sliding-Lid Simulation')
        plt.title('{}x{} size, {} time steps, {} omega, and {} wall speed'.format(Nx,Ny,timestep,omega,wall_vel))     
        plt.streamplot(xx, yy, ux_calc.T, uy_calc.T)
        plt.savefig('Images/MS7 Velocity Profile (streamplot)' + str(i),bbox_inches='tight')

print('Runtime: ',time.time()-starttime) #compute the total runtime
