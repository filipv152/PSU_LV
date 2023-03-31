import numpy
import numpy as np
from matplotlib import pyplot as plt

arr=[]
np.random.seed(104)
for x in range(100):
    x=np.random.randint(1,7)
    arr.append(x)

mat = numpy.array(arr)
print(mat)

plt.hist(mat,bins=(6))
plt.ylim(ymin=0, ymax= 100)
plt.title("Histogram")
plt.show()