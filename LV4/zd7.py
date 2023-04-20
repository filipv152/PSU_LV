from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Podjela na train i test skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skaliranje ulaznih podataka
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Kreiranje Ridge modela
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

# Treniranje modela na train skupu
ridge.fit(X_train, y_train)

# Predikcija na train i test skupu
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)

# Evaluacija modela
print("Train R^2 score:", r2_score(y_train, y_train_pred))
print("Test R^2 score:", r2_score(y_test, y_test_pred))
print("Train MAE:", mean_absolute_error(y_train, y_train_pred))
print("Test MAE:", mean_absolute_error(y_test, y_test_pred))
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
