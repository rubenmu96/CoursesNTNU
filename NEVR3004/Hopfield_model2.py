import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import random

def Weight_matrix(N,n_patt,patts):
    tmp = np.zeros((N,N))
    W = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i != j:
                for p in range(n_patt):
                    tmp[i,j] = tmp[i,j] +  patts[p,i]*patts[p,j]
    W = tmp/N
    return W

def Energy_eq(W, S):
    return - 0.5*np.matmul(np.matmul(S.T, W), S)

def Hopfield(N,n_patt,T,noise_prob,noise_pattern):
    # Initialize parameters
    S = np.zeros(N)
    Energy = []
    mP = np.zeros((n_patt,T))
    patts = np.zeros((n_patt,N))
    update_inds = np.matlib.repmat(range(N),1,100)
    t = 0
    
    # Create random patterns
    for j in range(n_patt):
        for i in range(N):
            patts[j,i] = np.random.choice((-1,1), p=[0.5,0.5])
    # Calculate weights and plot
    W = Weight_matrix(N,n_patt,patts)
    plt.imshow(W)
    plt.show()

    # Add noise to one of the patterns
    rand_inds = random.sample(range(0,N),int(noise_prob*N))
    S[:] = patts[noise_pattern,:]
    
    for i in rand_inds:
        if random.uniform(0,1) < 0.5:
            S[i] = -1
        else:
            S[i] = 1
    
    while t < T:
        for p in range(n_patt):
            mP[p,t] = np.dot(S,patts[p,:])/N

        h = np.dot(S,W)

        ind = update_inds[0][t]
        S[ind] = np.sign(h[ind])

        Energy.append(Energy_eq(W,S))
        t = t+1
    # Plot overlap
    plt.figure()
    plt.subplot(1,2,1)
    for i in range(n_patt):
        plt.plot(mP[i,:])

    # Plot energy
    plt.subplot(1,2,2)
    plt.plot(Energy)
    plt.show()
        
    
Hopfield(70,3,100,0.7,0)



