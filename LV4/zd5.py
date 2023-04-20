import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error

# ucitavanje ociscenih podataka
df = pd.read_csv('cars_processed.csv')
df = df.drop(['name'], axis=1)

# podijeli podatke na train i test set
X_train, X_test, y_train, y_test = train_test_split(df.drop('selling_price', axis=1), df['selling_price'], test_size=0.2, random_state=42)

# skaliranje ulaznih podataka
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# određivanje parametara linearnog regresijskog modela
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# evaluacija modela na trening setu
y_train_pred = model.predict(X_train_scaled)
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
max_error_train = max_error(y_train, y_train_pred)

print(f'Train set MAE: {mae_train:.2f}')
print(f'Train set MSE: {mse_train:.2f}')
print(f'Train set R2 score: {r2_train:.2f}')
print(f'Train set Max Error: {max_error_train:.2f}')

# evaluacija modela na test setu
y_test_pred = model.predict(X_test_scaled)
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
max_error_test = max_error(y_test, y_test_pred)

print(f'Test set MAE: {mae_test:.2f}')
print(f'Test set MSE: {mse_test:.2f}')
print(f'Test set R2 score: {r2_test:.2f}')
print(f'Test set Max Error: {max_error_test:.2f}')



#6.
# Promjena broja ulaznih varijabli može značajno utjecati na performanse modela na testnom skupu.
# Ako se ukloni neka važna varijabla, to može uzrokovati prenaučenost (overfitting) modela, 
# što znači da će performanse modela na trening skupu biti dobre, ali će loše raditi na novim primjerima. 
#S druge strane, ako se uključi neka nepotrebna varijabla, to može dovesti do podnaučenosti (underfitting) modela, što znači da će performanse modela biti loše na oba skupa. 
# Stoga je bitno odabrati optimalan broj ulaznih varijabli za model.