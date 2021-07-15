import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv("data.csv")
length=len(data['x[n]'])
X=data['x[n]']
Y=data['y[n]']

plt.figure()

plt.subplot(2,1,1)
plt.title("original temperature record")
plt.stem(range(length),X)

plt.subplot(2,1,2)
plt.title("Distorted temperature record")
plt.stem(range(length),Y)
plt.show()

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
#knl=np.array([1,4,6,4,1])/16
#convolvedVals=convolve(Y,knl)
plt.figure()
for i in range(1,6):
    convolvedVals=deNoise(Y,i)
    plt.subplot(5,1,i)
    plt.stem(range(length),convolvedVals)
plt.show()
