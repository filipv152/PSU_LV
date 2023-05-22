import numpy as np
from tensorflow import keras
from tensorflow.python.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# Prikazi nekoliko slika iz train skupa
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.axis('off')
plt.show()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")

# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

# Kreiraj model
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Definiraj karakteristike procesa ucenja
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# Provedi ucenje mreze
model.fit(x_train_s, y_train_s, batch_size=128, epochs=3, verbose=1)

# Izračunaj točnost na skupu podataka za učenje i testiranje
train_loss, train_acc = model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=0)
print('Train accuracy:', train_acc)
print('Test accuracy:', test_acc)

# Izradi predikcije na skupu podataka za testiranje
test_predictions = model.predict(x_test_s)
test_predictions = np.argmax(test_predictions, axis=1)
test_true_labels = np.argmax(y_test_s, axis=1)

# Izradi matricu zabune
confusion_mat = confusion_matrix(test_true_labels, test_predictions)
print('Confusion matrix:')
print(confusion_mat)

# Spremi model na disk
model.save_weights("mnist_model_weights.h5")
