import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
from utils import *

'''
The section of code below is used for reading the input from csv File.
'''
data=pd.read_csv("data.csv")
length=len(data['x[n]'])
X=data['x[n]']
Y=data['y[n]']

'''
The section of the code below contains metrics used to evaluate our models.
'''
def MSE(x):
    return np.mean((X-x)**2)

def DeltaJmp(x):
    return np.abs(np.diff(x))
def DeltaMetric(x):
    return DeltaJmp(x)[96]
'''
This is the function which returns that value of F(I)/F(h) for a particular value of angular frequency w.Because F(h) can attain
values very close/equal to 0 we have decided to cap it at a certain threshold value so that is approximately resembles F(I)/F(h).
More details are given in the Report.
'''
signal=deNoise(Y)# This is the signal which we are trying to deblurr, It is a global variable can be reset later
def sig(w,T=0.7):
    global signal
    denom=BlurFourier(w)
    if denom<T:
        denom+=T
        #denom=T
        #denom=1+T
    return Fourier(signal,w)/denom

'''
For performance improvements we precompute the signal values and store it in sig2 list. This drastically brings down
the running time.
'''
sig2=[]
def sig2C(N,T=0.7): #0.7 

    for r in tqdm(range(N)):
        sig2.append(sig(2*r*math.pi/N,T=T))
'''
The section of code below gets the deblurred signal.
Params:
    blurred : the blurred signal which needs to be deblurred. It is an array
    Part : Number of partitions of riemann sum to be taken. By default it is set to 193
'''
def deBlur(blurred,Part=193):
    global sig2
    global signal
    global length
    sig2=[]
    signal=blurred #resetting the global variable signal 
    sig2C(Part) #precomputing sig2
    deblurred=np.zeros(length)
    for i in tqdm(range(length)):
        deblurred[i]=Inv(sig2,i,N=Part) #ith index of deblurred corresponds to INV fourier transform for index i of the signal sig
    return deblurred
'''
The section of the code below fetches the signals X1 and X2 as asked in the question.
'''
sig2=[]
x1=deBlur(deNoise(Y,n=5))
plt.figure()
plt.subplot(2,1,1)
plt.title("X1")
#plt.plot(range(length),X)
#plt.plot(range(length),x1) # uncomment this part for seeing line plot of x1 
plt.stem(range(length),x1)
plt.legend(["X","X1"],loc="upper right")

plt.subplot(2,1,2)
plt.title("X2")
x2=deNoise(deBlur(Y),n=5)
#plt.plot(range(length),X)
#plt.plot(range(length),x2) # uncomment this part for seeing line plot of x2 
plt.stem(range(length),x2)
plt.legend(["X","X2"],loc="upper right")
plt.show()
'''
The section of code below displays the metrics for x1,x2 and Y.
'''
print("The mse of x1 is "+str(MSE(x1)))
print("The mse of x2 is "+str(MSE(x2)))
print("The mse of Y is "+str(MSE(Y)))
print("---")
print("The DeltaMetric for x1 is "+str(DeltaMetric(x1)))
print("The DeltaMetric for x2 is "+str(DeltaMetric(x2)))
print("The DeltaMetric for Y is "+str(DeltaMetric(Y)))
