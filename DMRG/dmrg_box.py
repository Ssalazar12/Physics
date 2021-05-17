import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib


#-------------------------------------------------------------------------------
# function definitions
#-------------------------------------------------------------------------------

def initialize_hamiltonian(size=4, periodic_boundary=False):
    # creates the initial discretized hamiltonian
    # args-> size:  Integer,  set to 4 as default
    # 		 periodic_boundary: Boolean. If False then only periodic boundaries
    # returns --> array representing the hamiltonian
    
    hamiltonian = np.asmatrix(np.zeros((size,size)))
    np.fill_diagonal(hamiltonian, 2)

    for i in range(0,size-1):
        # fill upper diagonal
        hamiltonian[i,i+1] = -1
        # fill lower diagonal
        hamiltonian[i+1,i] = -1
    
    # for periodc boundaries add -1s at opposite corners
    if periodic_boundary==True:
        hamiltonian[-1,0] = -1
        hamiltonian[0,-1] = -1     
     
    return hamiltonian

def update_hamiltonian(current_H, new_t12, new_h11, new_t34, new_h44):
    # updates the block hamiltonian to include the new matrix elements of the new basis
    # assumes that we have reflection symmetry so only 2 elements are needed
    # args---> current_H: array. Matrix representing the block hamiltonian from the last step
    #       new_t12, new_h11... : scalars representing block matrix elements on the new basis, 
    # returns---> new_H: Matrix representing the new hamiltonian
    
    # System
    new_H = current_H.copy()
    new_H[0,0] = new_h11
    new_H[0,1] = new_t12
    new_H[1,0] = new_t12
    # environment
    new_H[2,3] = new_t34
    new_H[3,2] = new_t34
    new_H[3,3] = new_h44
    
    return new_H

def find_ground_state(hamiltonian):
    # diagonalize and get the ground state energy and vector
    # args-> hamiltonian: matrix to diagonalize
    # returns -> ground_state eigen value as scalar and ground state eigen vector as array

    # find the eigen vecots and eigen values
    eigen_values, eigen_vectors = la.eig(np.asmatrix(hamiltonian))
    # find the ground state es the state with the minimum eigen value
    ground_index = eigen_values.argmin()
    E_ground_state = np.real(eigen_values[ground_index])
    psi_ground_state = eigen_vectors[:,ground_index]
    
    if np.sum(psi_ground_state) < 0:
        psi_ground_state = np.multiply(psi_ground_state,-1)
    
    return np.real(E_ground_state), psi_ground_state

def update_basis(psi, H_bloc, Tblock ,is_right=False):
    # performs the basis projection to update superblock hamiltonian according to right or left block
    # args-> psi: array for ground state wave function. 
    #        H_block, Tblock: number representing the blocks that we want to update, if is_right=True
    #               they corrspond to environment, else they correspond to the system        
    #       is_right: Boolean,indicates wether right or left block
    # returns-> Hnew, Tnew, scalars representing the new components of the superblock
    
    # indices for the projection corresponding to the type of block
    first_index = 0
    second_index = 0
    new_basis = 0
    a1 = 0
    a2 = 0
    
    # if right block we take the last two indices
    if is_right:
        first_index = 2
        second_index = 3
        # normalize
        new_basis = 1.0/np.sqrt(psi[first_index]**2 + psi[second_index]**2) * \
                                    np.asmatrix([psi[first_index],psi[second_index]])
        a1 = new_basis[0,0]
        # note that a2 is the updated wave function at site i+1
        a2 = new_basis[0,1]
        # calculate the new blocks
        Hnew = (a2**2)*H_bloc + 2*(a1**2) + 2*a1*a2*Tblock
        Tnew = -a1
    
    else:
        first_index = 0
        second_index = 1
        # normalize
        new_basis = 1.0/np.sqrt(psi[first_index]**2 + psi[second_index]**2) * \
                                    np.asmatrix([psi[first_index],psi[second_index]])
        a1 = new_basis[0,0]
        # note that a2 is the updated wave function at site i+1
        a2 = new_basis[0,1]
        # calculate the new blocks
        Hnew = (a1**2)*H_bloc + 2*(a2**2) + 2*a1*a2*Tblock
        Tnew = -a2
    
    return Hnew, Tnew, [a1,a2]

#-------------------------------------------------------------------------------
#  Exact diagonalization
#-------------------------------------------------------------------------------

sistem_sizes = [16,32,64,100,128,256,512,1024]
# arrays for open boundaries
open_energies = []
open_states = []
exact_open_energies = []
# arrays for periodic boundaries
periodic_energies = []
periodic_states = []
exact_periodic_energies = []

for i in range(0,len(sistem_sizes)):
    # first for open boundary conditions
    h_open = initialize_hamiltonian(sistem_sizes[i])
    # now for periodic boundary conditions
    h_periodic = initialize_hamiltonian(sistem_sizes[i],periodic_boundary=True)
    # diagonalize the Hamiltonian and find ground state for both boundaries
    e_open, psi_open = find_ground_state(h_open)
    open_energies.append(e_open)
    open_states.append(psi_open)
    exact_open_energies.append(np.pi**2/sistem_sizes[i]**2)
    
    # for periodic
    e_per, psi_per = find_ground_state(h_periodic)
    periodic_energies.append(e_per)
    periodic_states.append(psi_per)


# plotting the results
matplotlib.rcParams.update({'font.size': 14.5})

# ground state energy
plt.figure(figsize=(9,7.5))
plt.title(r"Ground State enery for Open Boundary Conditions", y=1.05)
plt.scatter(sistem_sizes,open_energies, label='Diagonalized Ground State Energies')
plt.scatter(sistem_sizes,exact_open_energies, marker = 'x', label='Analytical Ground State Energies')

# adjusting the plot
plt.xlabel(r'L', fontsize=15)
plt.ylabel(r'Energy, $[\frac{\hbar^2}{2m\Delta x^2}]$',fontsize=16)
plt.tight_layout()
plt.legend()
plt.savefig('exact_diagonalization_obc')

# calculate and plot the fractional error as a function of system size
p_error = np.abs(np.asarray(open_energies) - \
          np.asarray(exact_open_energies))/np.abs(np.asarray(exact_open_energies))

plt.figure(figsize=(9,7.5))
plt.title(r"Fractional Error for Open Boundary Conditions", y=1.05)
plt.plot(sistem_sizes,p_error)
plt.xlabel(r'L', fontsize=15)
plt.ylabel(r'Fractional Error',fontsize=16)
plt.savefig('error_obc')


# ground state energy
plt.figure(figsize=(9,7.5))
plt.title("Ground State enery for Periodic Boundary Conditions", y=1.05)
plt.scatter(sistem_sizes,periodic_energies)
plt.xlabel(r'L', fontsize=15)
plt.ylabel(r'Energy, $[\frac{\hbar^2}{2m\Delta x^2}]$',fontsize=16)
plt.tight_layout()
plt.legend()

plt.savefig('exact_diagonalization_pbc')


# exact energy for L=100
Exact_E = exact_open_energies[3]

print()
print("calculated E for L=100:", open_energies[3],"exact E: ",exact_open_energies[3],
      "error for L=100: ", p_error[3])

print()
print("calculated E for L=1024:", open_energies[-1],"exact E: ",exact_open_energies[-1],
      "error for L=1024: ", p_error[-1])

print()

#-------------------------------------------------------------------------------
# DRMG FOR PARTICLE IN A BOX
#-------------------------------------------------------------------------------

# System size definition
L = 100
# vectors that save effective hamiltonians and other operators
H_list = list(np.zeros(int(L), dtype=np.float64)) # superblock hamiltonian
all_h = np.zeros(int(L),dtype=np.float64)
all_t = np.zeros(int(L),dtype=np.float64)
L_basis = np.zeros(int(L),dtype=np.float64)
R_basis = np.zeros(int(L),dtype=np.float64)

# save ground state energies
E_list = []

# initialize the hamiltonian for the warmup phase
H = initialize_hamiltonian()
# initialize the blocks we only need 2 since we have reflection symmetry
H11 = 2.0
T12 = -1.0

all_h[0] = H11
all_t[0] = T12
L_basis[0] = -1.0     
R_basis[-1] = -1.0

for i in range(0,int(L/2)-1):
    # diagonalize and get the ground state energy and vector
    E_ground, psi_ground = find_ground_state(H)
    # calculate the matrix element
    H11, T12, new_basis = update_basis(psi_ground, H11, T12 ,is_right=False)
               
    # update the hamiltonian and get the next site hamiltonian
    H = update_hamiltonian(H, T12, H11, T12, H11)
    
    # save the effective hamiltonian and other operators
    H_list[i+1] = H
    all_h[i+1] = H[0,0]
    all_t[i+1] = H[0,1]
               
    # Calculate L_l+1 block and it's reflection for the R_l+3 block
    L_basis[:i+1] = L_basis[:i+1]*new_basis[0]
    L_basis[i+1] = new_basis[1]
    
    E_list.append(E_ground)
    
# we can now reflect the left basis to create the right basis
R_basis = np.flip(L_basis)
infinite_psi = L_basis.copy()


err_ = np.abs(E_list[-1] - Exact_E)/np.abs(Exact_E)
print("Warm-up Energy: ", E_list[-1], "Warm-up Fractional error: ", err_)

# FINITE ALGORITHM PHASE-------------------------------------------------------

# number of sweeps
N_sweeps = 5
H_super = initialize_hamiltonian()

# save the finite system data for comparisson
sweep_h11 = all_h.copy()
sweep_t12 = all_t.copy()
sweep_h44 = np.zeros(len(sweep_h11),dtype=np.float64)
sweep_t34 = np.zeros(len(sweep_h11),dtype=np.float64)

sweep_energies =[]
sweep_psi = []

# one sweeps is right to left and left to right
for k in range(0, N_sweeps):
    # apply reflection symmetry
    sweep_h44[int(L/2)+1] = sweep_h11[int(L/2)-2]
    sweep_t34[int(L/2)+1] = sweep_t12[int(L/2)-2]
    
    # right to left
    for i in range(int(L/2)+1,2,-1):
        # form the superblock H starting from the middle and sweeping to the left 
        H_super = update_hamiltonian(H_super, sweep_t12[i-3], sweep_h11[i-3], sweep_t34[i], sweep_h44[i])
        # find ground state and update basis for the RIGHT block
        E_ground, psi_ground = find_ground_state(H_super)
        H44, T34, new_basis = update_basis(psi_ground, H_super[3,3], H_super[2,3], is_right=True)

        # update the chain
        sweep_h44[i-1] = H44
        sweep_t34[i-1] = T34
        # update the basis blocks
        R_basis[i-1:] = R_basis[i-1:]*new_basis[1]
        R_basis[i-1] = new_basis[0]
                
    # left to right
    for i in range(0, int(L/2)-2):
        # form the superblock H starting from the middle and sweeping to the left    
        H_super = update_hamiltonian(H_super, sweep_t12[i], sweep_h11[i], sweep_t34[i+3], sweep_h44[i+3])
        # find ground state and update basis for the RIGHT block
        E_ground, psi_ground = find_ground_state(H_super)
        H11, T12, new_basis = update_basis(psi_ground, H_super[0,0], H_super[0,1], is_right=False)

        # update the chain
        sweep_h11[i+1] = H11
        sweep_t12[i+1] = T12
        L_basis[:i+1] = L_basis[:i+1]*new_basis[0]
        L_basis[i+1] = new_basis[1]
        
        
    sweep_psi.append(list(L_basis))
        
    sweep_energies.append(E_ground)
    
    print()
    print('Sweeps Finished:', k)
    print()

err_ = np.abs(sweep_energies[-1] - Exact_E)/np.abs(Exact_E)
print("Final DMRG Energy: ", sweep_energies[-1], "Warm-Final Fractional error: ", err_)



# PLOTTING RESULTS-------------------------------------------------------------------------

L_list = np.linspace(0,L,L)
q = np.pi/(L+1)
true_psi = np.sin(q*L_list) * np.sqrt(4/L)

plt.figure(figsize=(9,7.5))
plt.title('Energy Convergence')
plt.plot(sweep_energies)

plt.xlabel("Number of Sweeps", fontsize=15)
plt.ylabel(r'Energy, $[\frac{\hbar^2}{2m\Delta x^2}]$',fontsize=16)
plt.savefig('energy_sweeps')

plt.figure(figsize=(11,9))
plt.title('Wave Function Convergence')

plt.plot(true_psi[:50], label = 'Exact $\psi$', color = 'red')

for i in range(0,N_sweeps):
    plt.plot(np.asarray(sweep_psi[i][:50]), label=i, linestyle='dashed')
    
plt.xlabel("Latice site ($l$)", fontsize=15)
plt.ylabel(r'$\psi$',fontsize=16)
plt.legend()

plt.savefig('psi_convergence')

plt.figure(figsize=(10,8.5))

for i in range(0,len(sweep_energies)):
    plt.hlines(sweep_energies[i],0,50, linestyles='dashdot')
    
plt.hlines(Exact_E,0,50, label='True E', color='orange')

plt.plot(E_list, label='Warmup', marker='x')
plt.ylim(-0.001,0.01)
plt.legend()

plt.ylabel(r'Energy, $[\frac{\hbar^2}{2m\Delta x^2}]$',fontsize=16)
plt.xlabel("Latice site ($l$)", fontsize=15)

plt.savefig('energy_comparison')













