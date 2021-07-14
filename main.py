import pandas as pd
import matplotlib.pyplot as plt

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
