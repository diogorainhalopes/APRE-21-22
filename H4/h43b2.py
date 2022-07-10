import matplotlib.pyplot as plt
import numpy as np

q = np.array([2,5,10,30, 100,300,1000])
fig, ax = plt.subplots()
plt.plot(q, 3*(q**2)+(5*q)+2, 'r', q, (q**2)+(3*q)+1, 'g')
ax.set_title('VCD with variable feature number')
ax.legend(labels=['MLP', 'Bayesian Classifier'], loc=2, fontsize = 9)

ax.set_xlabel('Data dimensionality')
ax.set_ylabel('VC dimension')
plt.show()