import numpy as np
import math
from tqdm import tqdm

'''
This function implements 1D kernel convolution.It takes two arguments 
1) an Array which contains the 1D signal.
2) The kernel over which the convolution has been implemented.

We have padded the edge cases with 0. Tho other values can be used to but 0 is a norm.
'''
def convolve(signal,kernel):
    kernel=kernel[::-1] #reversing the kernel for sliding window sum
    N=len(signal) # N is the length of the 1D signal
    M=len(kernel) # M is the length of the 1D kernel
    output=np.zeros(N) # we initialise the output by keeping its length to N

    for n in range(N):
        accumulated=0  # The value of this variable will hold the nth value of the output
        for i in range(M):        
            shifted_i=i-(M//2) # The origin of the kernel usually is assumed to be the mid value
                               # For e.g in 1/16[1 4 6 4 1] kernel we usually take the value 6 as the origin so the 
                               # so we evaluate the weighted sum at position n in the signal with placing the kernel's
                               # origin value at n and then taking the weighted sum 
            if n+shifted_i>=0 and n+shifted_i<N: #This if condition ensures the indices are within bounds and deals with the edge conditions.
               accumulated+=signal[n+shifted_i]*kernel[i] 
            else:                               
                accumulated+=signal[n]*kernel[i] #padding used to avoid edge effects
        output[n]=accumulated
    return output

'''
This function deNoises a signal by using an averaging kernel of the form [1/n 1/n ... 1/n] whose length is n.
By default the value of n is taken to be 3 and the kernel then is [1/3 1/3 1/3] kernel.
'''
def deNoise(signal,n=3):
    return convolve(signal, np.ones(n)/n) 

'''
This is the Fourier Transform function for the given blur impulse response 1/16[1 4 6 4 1]. The evaluation of the same 
is mentioned in the report. This is merely done to improve the speed, It is possible to pass the kernel in Fourier() function 
and obtain the Fourier transform value.
'''
def BlurFourier(w):
    return (np.cos(w/2))**4

'''
This is a Fourier transform function. It takes in two parameters
1) The 1D signal array of which the FT is to be computed 
2) A number w which specifies the value of angular frequency at which the Fourier is to be computed
'''
def Fourier(signal,w):
    ans=0
    for n in range(len(signal)):
        expo=np.complex(0,-w*n) 
        ans+=signal[n]*np.exp(expo) # adding the term X(jw)*e^(-jwn)
    return ans

'''
This is an Inverse Fourier transform function. It takes in two parameters
1) An array signal which represents X(jw) and signal(r) should give the value of X(jw) at w=2piR/N
2) A number n which specifies the value of index at which the Inverse Fourier is to be computed
3) It takes a number N which specifies the number of terms to be taken in riemann integration.The 
   more this number is the more accurate the outcome will be at the cost of computation time.
   By default it is set to 50
'''

def Inv(signal,n,N=50):
    ans=0
    for r in range(N):
        w=r*2*math.pi/N
        expo=np.complex(0,w*n)
        ans+=signal[r]*np.exp(expo)
        #ans+=signal(w)*np.exp(expo)
        #ans+=signal(w)*np.cos(w*n)
    return ans/N
