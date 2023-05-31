import os
import shutil
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Input, MaxPooling2D, Conv2D, Dense, Dropout, Flatten
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
import tensorflow as tf
import seaborn as sns

train_ds = image_dataset_from_directory(
    directory='Train/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(48, 48)
)

test_ds = image_dataset_from_directory(
    directory='Train/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(48, 48)
)

input_mask = (48,48,3)
inputs = Input(input_mask)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size = (3,3), activation="relu"))
model.add(Conv2D(filters=32, kernel_size = (3,3), activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3), activation = "relu"))
model.add(Conv2D(64,(3,3), activation = "relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(43, activation = "softmax"))

model.compile(loss="categorical_crossentropy", optimizer = "adam", metrics = "accuracy")

model.fit(train_ds, epochs = 5)
metrics = model.evaluate(test_ds,verbose=2)

print("Loss:",metrics[0])
print("Accuracy:",metrics[1])

y_predicted = model.predict(test_ds)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=test_ds, predictions=y_predicted_labels)
plt.figure(figsize = (10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

print(model.evaluate(test_ds))