import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib
import itertools

#------------------------------------------------------------------------
# definition of the operators
#------------------------------------------------------------------------

s_up = np.asarray([[0,1],[0,0]])
s_down = np.asarray([[0,0],[1,0]])
s_z = np.asarray([[1/2,0],[0,-1/2]])

#------------------------------------------------------------------------
# function definitions
#------------------------------------------------------------------------

def apply_operator(current_operator, next_operator,spin_chain, position):
    # applies the specified operators to the current and next place in the chain
    # args-> current_operator, next_operator: matrices representing the action on the current and 
    #        the next lattice site
    #         spin_chain: list of lists representing a basis vecotr
    #         position: index indicating the current position of the lattice
    # returns the transoformed spin chain
    spins_f = spin_chain.copy()
    # matrix multiplication only in the correct position of the basis vector
    spins_f[position] = list(np.matmul(current_operator,spin_chain[position]))
    spins_f[position+1] = list(np.matmul(next_operator,spin_chain[position+1]))    
    
    return spins_f

def calculate_matrix_term(spin_bra, spin_ket):
    # applies the heisenberg hamiltonian to each lattice site, for the hesinberg hamiltonian
    # we have 3 terms for each site.
    # args-> spin_bra: a list of lists representing the bra basis vector,
    #        spin_ket: a list of lists representing the ket basis vector
    # returns the matrix element as a number
    
    # save the spin chains just in case we need them latter
    sz_term = []
    # ket acted upon by first ladder operator product
    first_ladder = []
    # ket acted upon by second ladder operator product
    second_ladder = []
    eigen_values = []
    for i in range(0,len(spin_ket)-1):
        #Sz operator term
        transformed_spins = apply_operator(s_z,s_z,spin_ket,i)
        sz_term.append(transformed_spins)

        # First ladder operator term 
        transformed_spins = apply_operator(s_up,s_down,spin_ket,i)
        first_ladder.append(transformed_spins)

        # second ladder operator term
        transformed_spins = apply_operator(s_down,s_up,spin_ket,i)
        second_ladder.append(transformed_spins)

        # now we take the inner product with the basis Bra
        # to represent the inner product, sum the rows and then multiply all the elements to get the eigen value
        bracket = np.multiply(spin_bra,sz_term[i])
        eigen_values.append(np.prod(bracket.sum(1)))
        
        # remember that ladder operator terms have a 1/2 in front of them
        bracket = np.multiply(spin_bra,first_ladder[i])
        eigen_values.append(0.5*np.prod(bracket.sum(1)))

        bracket = np.multiply(spin_bra,second_ladder[i])
        eigen_values.append(0.5*np.prod(bracket.sum(1)))

    return np.sum(eigen_values)

def calculate_hamiltonian(system_basis_list,):
    # calculates each of the matrix terms of the hamiltonian
    # args-> basis_list: a list containing the basis states
    # returns-> h: array representing the hamiltonian
    h = np.zeros((len(system_basis_list), len(system_basis_list)))
    # columns iteration
    for i in range(len(system_basis_list)):
        # row iteration
        for j in range(len(system_basis_list)):
            h[j,i] = calculate_matrix_term(system_basis_list[j],system_basis_list[i])
    return h

def calculate_ground_state_matrix(system_basis, environment_basis, chosen_state_basis, c_eigen_vector):
    # Calculates the matrix representation of the groundstate wave functions
    # args-> system_basis: list containing the basis for the left block, 
    #        environment_basis: list containing the basis for the right block
    #        chosen_state_basis: list contatining the basis of the chosen state FOR THE WHOLE CHAIN
    #        c_eigen_vector: array with the eigenvector for the chosen state (ground state in this case)
    # returns -> psi_ij: array for the matrix representation of the groundstate
    psi_ij = np.zeros((len(environment_basis), len(system_basis)))
    # rows
    for i in range(0,len(environment_basis)):
        # columns
        for j in range(0,len(system_basis)):
            is_in_gbasis = system_basis[j]+environment_basis[i] in chosen_state_basis
            # ask if formed state is part of the ground state of the chain
            if is_in_gbasis == True: 
                # save index of the basis vector to find the eigen value
                eigen_index = basis_list.index(system_basis[j]+environment_basis[i])
                psi_ij[i,j] = c_eigen_vector[eigen_index]
                
    return psi_ij


#------------------------------------------------------------------------
# HAMILTONIAN DIAGONALIZATION AND GROUND STATE MATRIX
#------------------------------------------------------------------------

# generates the basis for the full L = 4 hamiltonian
basis_full_h = []
for comb in itertools.product([[1,0],[0,1]], repeat=4):
    basis_full_h.append(list(comb))
    
# calculates the complete hamiltonian
full_hamiltonian = calculate_hamiltonian(basis_full_h)

# diagonalization
v, _ = la.eig(full_hamiltonian)
print("eigen values for L=4 hamiltonian")
print(v)

# Now we have to calculate all the matrix elements 
#for the ground state part of the Hamiltonian. The ground state is given by all the configurations corresponding to S=0

# Defining the ground state basis
basis_1 = [[1,0],[1,0], [0,1], [0,1]]
basis_2 = [[1,0],[0,1], [1,0], [0,1]]
basis_3 = [[1,0],[0,1], [0,1], [1,0]]
basis_4 = [[0,1],[1,0], [1,0], [0,1]]
basis_5 = [[0,1],[1,0], [0,1], [1,0]]
basis_6 = [[0,1],[0,1], [1,0], [1,0]]

basis_list = [ basis_1, basis_2, basis_3, basis_4, basis_5, basis_6]

ground_state_hamiltonian = calculate_hamiltonian(basis_list)
ground_state_hamiltonian

# matrix diagonalization
# .eig returns a tuple of vectors
g_eigenvalues, g_eigenvectors = la.eig(np.asmatrix(ground_state_hamiltonian))
# save the lowest eigenvalue and eigenvector which represent the ground state
ground_index = g_eigenvalues.argmin()

# PRINTING RESULTS
print("Eigenvalues for the groundstate block")
print(g_eigenvalues)
print()

# We now choose the lowest eigen value since it corresponds to the ground state
print("chosen eigenvalue")
print(g_eigenvalues[ground_index])
print()
print("chosen eigenvector")
print(g_eigenvectors[:,ground_index])
chosen_eigenvector = g_eigenvectors[:,ground_index]


#------------------------------------------------------------------------
# REDUCED DENSITY MATRIX FOR 2 SITES
#------------------------------------------------------------------------

print("//////////////////---------------------------//////////////////")
print("Calculating the reduced density matrix for 2 sites")
print()

# define the two system basis
A_1 = [[1,0],[1,0]]
A_2 = [[1,0],[0,1]]
A_3 = [[0,1],[1,0]]
A_4 = [[0,1],[0,1]]

A_basis = [A_1,A_2,A_3,A_4]
B_basis = [A_1,A_2,A_3,A_4]

np.set_printoptions(precision=7)
# PRINTING RESULTS

ground_matrix = calculate_ground_state_matrix(A_basis, B_basis, basis_list, chosen_eigenvector) 
print("ground state matrix")
print(ground_matrix)
print()
print("Reduced density matrix")
rho_reduced = np.asmatrix(ground_matrix)* np.asmatrix(ground_matrix).H
print(rho_reduced)
print()
print("eigen values of the reduced density matrix")
_eigenvalues, _ = la.eig(rho_reduced)
print(_eigenvalues)
print()

#------------------------------------------------------------------------
# REDUCED DENSITY MATRIX FOR 1 SITE
#------------------------------------------------------------------------

A_basis = [[[1,0], [0,1]]]
B_basis = [A_1,A_2,A_3,A_4]

print("//////////////////---------------------------//////////////////")
print("Calculating the reduced density matrix for 1 sites")
# PRINTING RESULTS
ground_matrix = calculate_ground_state_matrix(A_basis, B_basis, basis_list, chosen_eigenvector) 
print("ground state matrix")
print(ground_matrix)
print()

print("Reduced density matrix")
rho_reduced = np.asmatrix(ground_matrix)* np.asmatrix(ground_matrix).H
print(rho_reduced)
print()

print("eigen values of the reduced density matrix")
_eigenvalues, _ = la.eig(rho_reduced)
print(_eigenvalues)
print()




