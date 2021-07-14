import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("data.csv")
length=len(data['x[n]'])
X=data['x[n]']
Y=data['y[n]']

plt.figure()

plt.subplot(1,2,1)
plt.title("original temperature record")
plt.stem(X,range(length))

plt.subplot(1,2,2)
plt.title("Distorted temperature record")
plt.stem(Y,range(length))
plt.show()
