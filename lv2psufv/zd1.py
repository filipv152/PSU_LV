import numpy as np
import matplotlib.pyplot as plt

a1 = np.array ([1,3,3,2,1])
a2 = np.array ([1,1,2,2,1])

plt.plot(a1,a2,linewidth=3,color='red')
plt.ylim(ymin=0, ymax=4)
plt.xlim(xmin=0, xmax=4)
plt.show()