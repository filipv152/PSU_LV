from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import color
import matplotlib.image as mpimg
import numpy as np

# Učitaj model
model = load_model("mnist_model_weights.h5")

filename = 'test.png'

img = mpimg.imread(filename)
img = color.rgb2gray(img)
img = resize(img, (28, 28))

plt.figure()
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()

img = img.reshape(1, 28, 28, 1)
img = img.astype('float32')

# Napravi predikciju
prediction = model.predict(img)
predicted_label = np.argmax(prediction)

# Ispiši rezultat
print("Predicted label:", predicted_label)
