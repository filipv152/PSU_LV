import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mtcars = pd.read_csv('C:\\Users\\student\\Desktop\\LV3 PSU\\mtcars.csv')
print(len(mtcars))
print(mtcars)
print('\n')

cilindri6 = mtcars[mtcars.cyl==6]['mpg'].mean()
cilindri4 = mtcars[mtcars.cyl==4]['mpg'].mean()
cilindri8 = mtcars[mtcars.cyl==8]['mpg'].mean()

graphdata=pd.DataFrame({'cilindri':['4','6','8'], 'pot':[cilindri4,cilindri6,cilindri8]})
ax = graphdata.plot.bar(x='cilindri', rot=0, title='Potrosnja')
plt.show()

boxplot = mtcars.boxplot(by='cyl', column=['wt'],boxprops=dict(color='blue'))
plt.show()
