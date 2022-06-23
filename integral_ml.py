import tensorflow as tf
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers


cols=["a", "b", "c", "y"]
dataset = pd.read_csv("./data.csv", usecols=cols)
for i in cols:
    dataset[i] = dataset[i].astype(float)

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
print(test_dataset.head())

train_labels = train_dataset.pop("y")
test_labels = test_dataset.pop("y")

print("\nTRAIN DATASET")
print(train_dataset.head())
print("\nTRAIN LABELS")
print(train_labels.head())
print("\nTEST DATASET")
print(test_dataset.head())
print("\nTEST LABELS")
print(test_labels.head())

normalizer = layers.Normalization()
normalizer.adapt(np.array(train_dataset))

model = tf.keras.Sequential([
    normalizer,
    layers.Dense(
        units=32,
        activation="sigmoid",
        input_dim=8
    ),
    layers.Dense(units=1)
])

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.003),
    loss="mean_squared_error"
)

start_learning_time = time.time()
history = model.fit(
    train_dataset,
    train_labels,
    batch_size=200,
    epochs=800,
    validation_split=0.2
)
end_learning_time = time.time()

test_results = model.evaluate(
    test_dataset,
    test_labels
)

print("\n=================================================================")
print("Time: ", end_learning_time-start_learning_time)
print("Loss: ", test_results)
print("=================================================================\n")
print(model.summary())


def plot_loss():
  plt.plot(history.history["loss"], "r--")
  plt.plot(history.history["val_loss"], "g--")
  plt.ylim([0, 1.5])
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(["train", "test"], loc="best")
  plt.grid(True)
  plt.show()
plot_loss()

def plot_predictions():
    test_predictions = model.predict(test_dataset).flatten()
    plt.axes(aspect="equal")
    plt.scatter(test_labels, test_predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    lims = [-1500, 1500]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.show()
plot_predictions()
