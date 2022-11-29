import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
import torch

def create_binary_aperture_arr(p):
    '''
    Inputs
        p: int. prime integer
        
    Output
        A: np.array. Binary aperture array 
    '''
    A = np.zeros((p,p)) # binary aperture array
    # Aperture function p. 4350 in Gottesman and Fenimore (1989)
    for i in range(0,p):
        for j in range(0,p):
            C_i = legendre_symbol(i,p)
            C_j = legendre_symbol(j,p)

            if i == 0:
                A[i,j] = 0
            elif (j == 0 and i != 0):
                A[i,j] = 1
            elif (C_i * C_j) == 1:
                A[i,j] = 1
            else:
                A[i,j] = 0
    return A

def create_decoding_arr(A):
    '''
    Inputs
        A: np.array. Binary aperture array
        
    Output
        G: np.array of same size as A. Decoding function 
    '''
    
    G = np.zeros_like(A) # initialize decoding function
    p = G.shape[0]
    
    # Decoding function p. 4350 in Gottesman and Fenimore (1989)
    for i in range(0,p):
        for j in range(0,p):

            if (i + j) == 0:
                G[i,j] = 1
            elif (A[i,j] == 1 and (i + j) != 0):
                G[i,j] = 1
            elif (A[i,j] == 0 and (i + j) != 0):
                G[i,j] = -1
                
                
    return G

def FFT_convolve(A, B, p=None):
    '''
    Compute convolution using FFT between A and B
    Inputs: 
        A,B: np.ndarrays
        
    Outputs:
        conv_AB: nd.array. Convolution between A and B
    '''
    # Check A and B are the same size
    if np.array(A).shape != np.array(B).shape:
        print(A.shape, B.shape)
        raise Exception("The arrays A and B are not the same shape")
    
    # Define p if it is not given already
    if p is None:
        p = A.shape[0]
        
    # Do convolution via FFT   
    fft_A = fft2(A)
    fft_B = fft2(B)
    conv_AB = np.real(ifft2(np.multiply(fft_A,fft_B)))
    conv_AB = np.roll(conv_AB, [int((p-1)/2),int((p-1)/2)], axis=(0,1))
    
    return conv_AB

def add_Gaussian_noise(image, mean=0, var=0.1):
    """
    Inputs:
        image: np.array of size [height, width]. Image to which Gaussian filter will be added
        mean:  Mean for Gaussian distribution
        var:   Variance for Gaussian distribution
        
    Outputs:
        noisy: Image with added Gaussian noise
    """
    
    row,col = image.shape
    
    # Calculate Gaussian filter
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    
    # Add Gaussian filter to image
    noisy = image + gauss
    return noisy

def get_Gaussian_filter(image, mean=0, var=0.1):
    """
    Inputs:
        image: np.array of size [height, width]. Provides dimensions for filter
        mean:  Mean for Gaussian distribution
        var:   Variance for Gaussian distribution
        
    Outputs:
        noisy: Gaussian filter
    """
    
    row,col = image.shape
    
    # Calculate Gaussian filter
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    
    return gauss    
    
# --------- HELPER FUNCTIONS
                
# copied from https://eli.thegreenplace.net/2009/03/07/computing-modular-square-roots-in-python on 10/05/22
def legendre_symbol(a, p):
    """ Compute the Legendre symbol a|p using
        Euler's criterion. p is a prime, a is
        relatively prime to p (if p divides
        a, then a|p = 0)

        Returns 1 if a has a square root modulo
        p, -1 otherwise.
    """
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls


# normalize 
def normalize(data):
    normalized_data = (data-np.min(data))/(np.max(data)-np.min(data))
    return normalized_data

def get_D(x):
    return torch.unsqueeze(torch.tensor(FFT_convolve(np.squeeze(x.numpy()), A,p)), 0)