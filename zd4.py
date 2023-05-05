import matplotlib.image as mpimg
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Učitavanje slike
image = mpimg.imread('example.png')

# Izdvajanje dimenzija slike
w, h, d = tuple(image.shape)

# Prebacivanje slike u jednodimenzionalni niz
image_array = np.reshape(image, (w * h, d))

# Primjena K-means algoritma na niz piksela
n_clusters = 10  # broj klastera za kvantizaciju boje
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(image_array)

# Kvantizacija boje slike
compressed_image_array = kmeans.predict(image_array)
compressed_image = np.reshape(compressed_image_array, (w, h))

# Izračun kompresije
original_size = w * h * d
compressed_size = n_clusters * (d + 1)  # broj klastera * (broj dimenzija + 1)
compression_ratio = original_size / compressed_size

# Prikaz slika
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Originalna slika')
ax2.imshow(compressed_image, cmap='gray')
ax2.set_title('Kvantizirana slika s {} klastera\nKompresijski omjer: {:.2f}'.format(n_clusters, compression_ratio))
plt.show()



# Potrebno je nekoliko sekundi da se kod izvrši!
# Ovim kodom se na prvoj slici prikazuje originalna slika, a na drugoj slika komprimirana korištenjem k-means 
# algoritma sa odabranim brojem klastera. Također se izračunava kompresijski omjer u odnosu na originalnu sliku, 
# uz pretpostavku da se svaki intenzitet boje može prikazati u 8 bita (256 razina sive) te da se centroidi klastera 
# spremaju u 24 bita (3 * 8 bita za R, G, B kanal).
# Kada mijenjamo broj klastera, možemo primijetiti kako se komprimirana slika mijenja i kako se mijenja kompresijski omjer. 
# Ako odaberemo premali broj klastera, izgubiti ćemo puno detalja i slika će izgledati popikselirano. 
# Ako odaberemo prevelik broj klastera, tada će se slika činiti vrlo glatkom i neće biti dovoljno različitih intenziteta boja.