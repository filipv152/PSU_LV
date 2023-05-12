from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import color
import matplotlib.image as mpimg
import numpy as np
import joblib

# Učitaj sliku i prikaži ju
filename = 'test.jpg'

img = mpimg.imread(filename)

# Ukloni alfa kanal ako postoji
if img.shape[2] == 4:
    img = img[:, :, :3]

img = color.rgb2gray(img)
img = resize(img, (28, 28))

plt.figure()
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()

# Prebaci sliku u vektor odgovarajuće veličine
img_vector = img.reshape(1, -1)

# Vrijednosti piksela kao float32
img_vector = img_vector.astype('float32')

# Učitavanje modela
filename = "NN_model.sav"
mlp_mnist = joblib.load(filename)

# Napravi predikciju
prediction = mlp_mnist.predict(img_vector)

print("------------------------")
print("Slika sadrži znamenku:", prediction[0])
