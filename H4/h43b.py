import matplotlib.pyplot as plt
import numpy as np

t = np.array([2,5,10,12,13])
fig, ax = plt.subplots()
plt.plot(t, 3*(t**2)+(5*t)+2, 'r', t, 3**(t), 'b', t, (t**2)+(3*t)+1, 'g--')
ax.set_title('VCD with variable feature number')
ax.legend(labels=['MLP', 'Decision Tree', 'Bayesian Classifier'], loc=2, fontsize = 9)

ax.set_xlabel('Data dimensionality')
ax.set_ylabel('VC dimension')
plt.show()