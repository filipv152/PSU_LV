import numpy as np
from sklearn.datasets import fetch_openml
import joblib
import pickle
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import itertools
import ssl

ssl._create_default_https_context=ssl._create_unverified_context

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Prikaz nekoliko ulaznih slika
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
    ax.set_title(f'Label: {y[i]}')
plt.tight_layout()
plt.show()

# Skaliranje podataka, train/test split
X = X / 255.
X_train, X_test = X[:10000], X[10000:]
y_train, y_test = y[:10000], y[10000:]

# Izgradnja vlastite mreže pomocu MLPClassifier
mlp_mnist = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)

# Treniranje modela
mlp_mnist.fit(X_train, y_train)

# Evaluacija modela
train_accuracy = mlp_mnist.score(X_train, y_train)
test_accuracy = mlp_mnist.score(X_test, y_test)

print(f"Točnost na skupu za učenje: {train_accuracy:.2f}")
print(f"Točnost na skupu za testiranje: {test_accuracy:.2f}")

# Funkcija za prikaz matrice zabune
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Stvarna vrijednost')
    plt.xlabel('Predviđena vrijednost')

# Prikaz matrice zabune na skupu za učenje
plt.figure(figsize=(8, 6))
y_train_pred = mlp_mnist.predict(X_train)
plot_confusion_matrix(y_train, y_train_pred, classes=np.unique)
