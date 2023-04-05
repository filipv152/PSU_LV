import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = plt.imread("tiger.png")
img = img[:,:,0].copy()

t=[]

t=img + 0.7
t[t>1]=1
plt.figure()
plt.imshow(t, cmap='gray')
plt.show()

t = np.rot90(t,3)
plt.figure()
plt.imshow(t, cmap='gray')
plt.show()

t=np.fliplr(t)
plt.figure()
plt.imshow(t, cmap='gray')
plt.show()

t= t[::10, ::10]
plt.figure()
plt.imshow(t, cmap='gray')
plt.show()

height, width = img.shape

new_img = np.zeros((height,width),dtype=np.float32)
new_width = int (width/4)
new_img[:, :new_width] = img[:, new_width:2*new_width]

plt.figure()
plt.imshow(new_img, cmap='gray')
plt.show()