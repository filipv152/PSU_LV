import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage

X, _ = make_blobs(n_samples=20, centers=5, random_state=42)

# Računanje poveznica između klastera pomoću metode 'ward'
Z = linkage(X, method='ward')

# Prikazivanje dendrograma
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.show()

# Dendrogram se može vizualizirati uz pomoć matplotlib biblioteke, a koristimo funkciju dendrogram iz modula scipy.cluster.hierarchy. 
# Podešavanjem argumenta method u funkciji linkage možemo mijenjati korištenu metodu za određivanje povezanosti između klastera. 
# Na primjer, ako postavimo method='single', koristit ćemo metodu jedne veze za određivanje povezanosti između klastera.
# Ovisno o odabranoj metodi i skupu podataka, dendrogram može prikazati različite strukture klastera. 
# U slučaju Zadatka 1, korištenje metode 'ward' (što je pretpostavljena vrijednost ako ne navedemo method argument) 
# daje jasnu strukturu s pet klastera. Međutim, korištenje druge metode poput 'single' ili 'complete' može dovesti 
# do nejasne strukture ili "zamršenog" dendrograma.
# Dakle, odabirom prave metode moguće je dobiti jasnu i korisnu strukturu klastera iz dendrograma.