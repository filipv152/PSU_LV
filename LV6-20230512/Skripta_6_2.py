from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import color
import matplotlib.image as mpimg
import numpy as np
import joblib

# ucitaj sliku i prikazi ju
filename = 'test.png'

img = mpimg.imread(filename)
img = color.rgb2gray(img)
img = resize(img, (28, 28))

plt.figure()
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


# TODO: prebacite sliku u vektor odgovarajuce velicine


# vrijednosti piksela kao float32
img = img.astype('float32')

# ucitavanje modela
filename = "NN_model.sav"
mlp_mnist = joblib.load(filename)

# TODO: napravi predikciju i spremi u varijablu digit kao string


print("------------------------")
print("Slika sadrzi znamenku: ", label)