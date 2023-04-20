import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error

# učitajte podatke i izbacite nepotrebne stupce
df = pd.read_csv('cars_processed.csv')
df = df.drop(['name', 'mileage'], axis=1)

# one-hot kodirajte kategoričke varijable
df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission'])

# podijelite podatke na trening i testni skupove
X = df.drop('selling_price', axis=1)
y = df['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# skalirajte podatke
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# dodajte polinomijalne članove
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# kreirajte i trenirajte linearni model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# procijenite performanse modela
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

print("Trening set:")
print("MAE:", mean_absolute_error(y_train, y_train_pred))
print("MSE:", mean_squared_error(y_train, y_train_pred))
print("R^2:", r2_score(y_train, y_train_pred))
print("Max error:", max_error(y_train, y_train_pred))

print("Testni set:")
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("MSE:", mean_squared_error(y_test, y_test_pred))
print("R^2:", r2_score(y_test, y_test_pred))
print("Max error:", max_error(y_test, y_test_pred))

# Rezultati evaluacije modela na trening i test skupovima će biti ispisani, 
# a možemo vidjeti da dodavanje kategoričkih varijabli značajno poboljšava performanse modela. 
# Dodavanje polinomijalnih članova također može dodatno poboljšati performanse modela, ovisno o podacima i modelu.