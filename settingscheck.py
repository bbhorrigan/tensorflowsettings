import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Create a small dataset
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Define a simple neural network
model = Sequential([
    Dense(2, input_dim=2, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1000, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X_train, y_train)
print("Loss:", loss)
print("Accuracy:", accuracy)
