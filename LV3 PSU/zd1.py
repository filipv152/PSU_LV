import pandas as pd
import numpy as np

mtcars = pd.read_csv('C:\\Users\\student\\Desktop\\LV3 PSU\\mtcars.csv')
print(len(mtcars))
print(mtcars)
print('\n')

print("Najefikasniji automobili: ")
print(mtcars.sort_values(by=['mpg']).tail(5))

print('\n')
print('Najefikasniji s 8 cilindara')
print(mtcars[mtcars['cyl']==8].sort_values(by=['mpg']).tail(3))

print('\n')
print('Srednja potrosnja 6 cilindricnih automobila je: ')
print(mtcars[mtcars.cyl==6]['mpg'].mean())

print('\n')
print('Srednja potrošnja automobila s 4 cilindra mase između 2000 i 2200 lbs')
print(mtcars[(mtcars.cyl==4) & (mtcars.wt > 2.000) & (mtcars.wt < 2.200)]['mpg'].mean())

print('\n')
print('Automobili s ručnim i automatskim mjenjačem u ovom skupu podataka')
print('Automatski prijenos: ',mtcars[mtcars.am==1]['am'].count())
print('Rucni prijenos: ',mtcars[mtcars.am==0]['am'].count())

print('\n')
print('Automobili s automatskim mjenjačem i snagom preko 100 konjskih snaga')
print(mtcars[(mtcars.am==1) & (mtcars.hp > 100)]['am'].count())

print('\n')
print('Masa svakog automobila u kilogramima')
mtcars['wt_kg'] = mtcars['wt']*0.45
print(mtcars[['car', 'wt_kg']])