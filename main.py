import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
data=pd.read_csv("data.csv")
length=len(data['x[n]'])
X=data['x[n]']
Y=data['y[n]']

#plt.figure()

#plt.subplot(2,1,1)
#plt.title("original temperature record")
#plt.stem(range(length),X)

#plt.subplot(2,1,2)
#plt.title("Distorted temperature record")
#plt.stem(range(length),Y)
#plt.show()

def convolve(signal,kernel):
    output=np.zeros(len(signal))
    for n in range(len(signal)):
        accumulated=0
        for i in range(len(kernel)):
            
            shifted_i=i-(len(kernel)//2)
            if n+shifted_i>=0 and n+shifted_i<len(signal):
               accumulated+=signal[n+shifted_i]*kernel[i] 
        output[n]=accumulated
    return output
def deNoise(signal,n=3):
    return convolve(signal, np.ones(n)/n)
def BlurFourier(w):
    return (np.cos(w/2))**4
def Fourier(signal,w):
    ans=0
    for n in range(len(signal)):
        expo=np.complex(0,-w*n)
        ans+=signal[n]*np.exp(expo)
    return ans
def Inv(signal,n):
    N=50
    ans=0
    for r in range(int(np.floor(N*2*math.pi))):
        expo=np.complex(0,r*n/N)
        ans+=signal(r/N)*np.exp(expo)
    return ans/(2*math.pi)
def sig(w):
    signal=X
    denom=BlurFourier(w)
    if denom<0.4:
        denom=0.4
    return Fourier(signal,w)/denom
deblurred=np.zeros(length)
for i in tqdm(range(length)):
    deblurred[i]=Inv(sig,i)
plt.figure()
plt.stem(range(length),deblurred)
#knl=np.array([1,4,6,4,1])/16
#convolvedVals=convolve(Y,knl)
#plt.figure()
#for i in range(1,6):
#    convolvedVals=deNoise(Y,i)
#    plt.subplot(5,1,i)
#    plt.stem(range(length),convolvedVals)
plt.show()
