import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from PIL import Image

# Učitavanje slike
image = np.array(Image.open('example.png'))
w, h, d = tuple(image.shape)
assert d == 3
image_array = np.reshape(image, (w * h, d))

# Kvantizacija
n_colors = 16
image_array_sample = shuffle(image_array, random_state=0, n_samples=1000)
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
labels = kmeans.predict(image_array)
image_quantized = np.reshape(labels, (w, h))

# Prikazivanje originalne i kvantizirane slike
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Originalna slika')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_quantized)
plt.title('Kvantizirana slika')
plt.axis('off')

plt.show()