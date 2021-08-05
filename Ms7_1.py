"""
HPC Milestone 7
Jared Engelken
Python 3
"""
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
from mpi4py import MPI


### Set Parameters, define grid & set initialization values
dtype = np.float32 #data type
timestep = 1000 #number of animation iterations
Nx = 300 #plot size in the x direction #15
Ny = 300 #plot size in the y direction #10
Y,X = np.meshgrid(range(Ny),range(Nx)) #create the plot grid

c = np.array([[0,  1,  0, -1,  0,  1, -1, -1,  1],    # velocities, x components
              [0,  0,  1,  0, -1,  1,  1, -1, -1]]).T # velocities, y components
weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) # weights for each channel
omega = 0.5  #relaxation rate #0.3 #A higher omega means a lower kinematic viscosity. Anything above 1.7 becomes unstable!
epsilon = 0.001 #pressure difference
wall_vel = 0.1 #wall speed
D = IntEnum('D', 'E N W S NE NW SW SE') #directions

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
    # Defining the boundary conditions --> Anti-direction for the bottom wall
    f[[2,5,6],:,0] = f_bottom[[4,7,8],:]    
    
    # Top boundary: sliding lid - compute rho after bounce back
    rho_bottom = f_top[6] + f_top[2] + f_top[5] + f[6,:,-1] + f[2,:,-1] + f[5,:,-1] + f[3,:,-1] + f[0,:,-1] + f[1,:,-1]
    f[4,:,-1] = f_top[4]
    f[8,:,-1] = f_top[6] + 6*weights[8]*rho_bottom*wall_vel
    f[7,:,-1] = f_top[5] - 6*weights[7]*rho_bottom*wall_vel
    
     # Change to "if False" for Couette flow test, otherwise set to "if True"
    if True:
        # Left boundary
        f[D.E,0,:] = f_left[D.W]
        f[D.NE,0,:] = f_left[D.SW]
        f[D.SE,0,:] = f_left[D.NW]

        # Right boundary
        f[D.W,-1,:] = f_right[D.E]
        f[D.NW,-1,:] = f_right[D.SE]
        f[D.SW,-1,:] = f_right[D.NE]

        # Bottom-left corner
        f[D.N,0,0] = f_bottom[D.S,0]
        f[D.E,0,0] = f_bottom[D.W,0]
        f[D.NE,0,0] = f_bottom[D.SW,0]

        # Bottom-right corner
        f[D.N,-1,0] = f_bottom[D.S,-1]
        f[D.W,-1,0] = f_bottom[D.E,-1]
        f[D.NW,-1,0] = f_bottom[D.SE,-1]

        # Top-left corner
        f[D.S,0,-1] = f_top[D.N,0]
        f[D.E,0,-1] = f_top[D.W,0]
        f[D.SE,0,-1] = f_top[D.NW,0]

        # Top-right corner
        f[D.S,-1,-1] = f_top[D.N,-1]
        f[D.W,-1,-1] = f_top[D.E,-1]
        f[D.SW,-1,-1] = f_top[D.NE,-1]


def communicate(f_ikl):
    """
    Communicate boundary regions to ghost regions.
    Parameters
    ----------
    f_ikl : array
        Array containing the occupation numbers. Array is 3-dimensional, with
        the first dimension running from 0 to 8 and indicating channel. The
        next two dimensions are x- and y-position. This array is modified in
        place.
    """
    # Send to left
    recvbuf = f_ikl[:, -1, :].copy()
    comm.Sendrecv(f_ikl[:, 1, :].copy(), left_dst,
                  recvbuf=recvbuf, source=left_src)
    f_ikl[:, -1, :] = recvbuf
    # Send to right
    recvbuf = f_ikl[:, 0, :].copy()
    comm.Sendrecv(f_ikl[:, -2, :].copy(), right_dst,
                  recvbuf=recvbuf, source=right_src)
    f_ikl[:, 0, :] = recvbuf
    # Send to bottom
    recvbuf = f_ikl[:, :, -1].copy()
    comm.Sendrecv(f_ikl[:, :, 1].copy(), bottom_dst,
                  recvbuf=recvbuf, source=bottom_src)
    f_ikl[:, :, -1] = recvbuf
    # Send to top
    recvbuf = f_ikl[:, :, 0].copy()
    comm.Sendrecv(f_ikl[:, :, -2].copy(), top_dst,
                  recvbuf=recvbuf, source=top_src)
    f_ikl[:, :, 0] = recvbuf

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
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
print('Size: ', size)
print('rank: ', rank)

ndx = int(np.sqrt(size))
ndy = int(np.sqrt(size))
print('Ndx = ', ndx)
print('Ndy = ', ndy)

if rank == 0:
    print('Running in parallel on {} MPI processes.'.format(size))
assert ndx*ndy == size
if rank == 0:
    print('Domain decomposition: {} x {} MPI processes.'.format(ndx, ndy))
    print('Global grid has size {}x{}.'.format(Nx, Ny))
    print('Using {} floating point data type.'.format(dtype))

# Create cartesian communicator and get MPI ranks of neighboring cells
comm = MPI.COMM_WORLD.Create_cart((ndx, ndy), periods=(False, False))
print('rank = {}/{}'.format(comm.Get_rank(), comm.Get_size()))

local_nx = Nx//ndx
local_ny = Ny//ndy
print("Local Nx = ", local_nx)
print("Local Ny = ", local_ny)

ux = np.zeros(local_nx)
ux.shape = (local_nx,1)
print('ux shape: ',ux.shape)

uy = np.zeros(local_ny)
uy.shape = (local_ny,1)
print('uy shape: ', uy.shape)

rho=np.ones((local_nx, local_ny))
print('Rho: ', rho.shape)

left_src, left_dst = comm.Shift(0, -1)
right_src, right_dst = comm.Shift(0, 1)
bottom_src, bottom_dst = comm.Shift(1, -1)
top_src, top_dst = comm.Shift(1, 1)

# We need to take care that the total number of *local* grid points sums up to
# nx. The right and topmost MPI processes are adjusted such that this is
# fulfilled even if nx, ny is not divisible by the number of MPI processes.
if right_dst < 0:
    # This is the rightmost MPI process
    local_nx = Nx - local_nx*(ndx-1)
without_ghosts_x = slice(0, local_nx)
if right_dst >= 0:
    # Add ghost cell
    local_nx += 1
if left_dst >= 0:
    # Add ghost cell
    local_nx += 1
    without_ghosts_x = slice(1, local_nx+1)
if top_dst < 0:
    # This is the topmost MPI process
    local_ny = Ny - local_ny*(ndy-1)
without_ghosts_y = slice(0, local_ny)
if top_dst >= 0:
    # Add ghost cell
    local_ny += 1
if bottom_dst >= 0:
    # Add ghost cell
    local_ny += 1
    without_ghosts_y = slice(1, local_ny+1)

mpix, mpiy = comm.Get_coords(rank)
print('Rank {} has domain coordinates {}x{} and a local grid of size {}x{} (including ghost cells).'.format(rank, mpix, mpiy, local_nx, local_ny))


### Calculation
f = equilibrium(rho, ux, uy)

### Plots
for i in range(timestep):
    communicate(f)
    bounce_back(f, wall_vel)
    rho_kl, ux_kl, uy_kl = collision(f, omega)

    if (i % 100 == 0 and i != 0) or i == timestep-1:
        save_mpiio(comm, 'ux_{}.npy'.format(i), ux_kl[without_ghosts_x, without_ghosts_y])
        save_mpiio(comm, 'uy_{}.npy'.format(i), uy_kl[without_ghosts_x, without_ghosts_y])
      
        ux_kl = np.load('ux_{}.npy'.format(i))
        uy_kl = np.load('uy_{}.npy'.format(i))
        
        nx, ny = ux_kl.shape
        
        plt.figure(figsize = (Nx+100,Ny+100)) #(10,10)
        x_k = np.arange(nx)
        y_l = np.arange(ny)
        plt.xlim((0,Nx))
        plt.ylim((0,Ny))
        plt.title('LBM Simulation with {}x{} size, {} time steps, and {} omega'.format(Nx,Ny,timestep,omega), pad = 50)
        plt.text(Nx+5,Ny/2,'right wall',color='blue',verticalalignment='center')
        plt.text(0-40,Ny/2,'left wall',color='blue',verticalalignment='center')
        plt.text(Nx/2-15,-15,'bottom wall',color='blue',verticalalignment='center')
        plt.text(Nx/2-15,Ny+10,'sliding lid',color='blue',verticalalignment='center')        
        plt.streamplot(x_k, y_l, ux_kl.T, uy_kl.T)
        #plt.show()
        plt.savefig('Images/MS7 Velocity Profile (streamplot)' + str(i).zfill(4))
