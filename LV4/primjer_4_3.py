import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ucitavanje ociscenih podataka
df = pd.read_csv('cars_processed.csv')
print(df.info())

# razliciti prikazi
sns.pairplot(df, hue='fuel')

sns.relplot(data=df, x='km_driven', y='selling_price', hue='fuel')
df = df.drop(['name','mileage'], axis=1)

obj_cols = df.select_dtypes(object).columns.values.tolist()
num_cols = df.select_dtypes(np.number).columns.values.tolist()

fig = plt.figure(figsize=[15,8])
for col in range(len(obj_cols)):
    plt.subplot(2,2,col+1)
    sns.countplot(x=obj_cols[col], data=df)

df.boxplot(by ='fuel', column =['selling_price'], grid = False)

df.hist(['selling_price'], grid = False)

tabcorr = df.corr()
sns.heatmap(df.corr(), annot=True, linewidths=2, cmap= 'coolwarm') 

plt.show()

#Zadatak 4:
#1. Dataset sadrži 301 izmjerenih automobila.
#2. Dataset sadrži stupce različitih tipova, uključujući float, int i object.
#3. Automobil s najvećom cijenom je Porsche Panamera, a automobil s najmanjom cijenom je Maruti 800.
#4. U datasetu je proizvedeno 61 automobila 2012. godine.
#5. Automobil koji je prešao najviše kilometara je BMW X5, a automobil koji je prešao najmanje kilometara je Maruti 800.
#6. Najčešće automobili imaju pet sjedala.
#7. Prosječna prijeđena kilometraža za automobile s dizel motorom je 76626.37 km, a prosječna prijeđena kilometraža za automobile s benzinskim motorom je 57693.43 km.