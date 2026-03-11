import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load dataset
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Step 2: Normalize data
print("Normalizing data...")
X_train = X_train / 255.0
X_test = X_test / 255.0

# Step 3: Build neural network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Step 4: Compile model
print("Compiling model...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train model
print("Starting training...")
model.fit(X_train, y_train, epochs=5)

# Step 6: Evaluate model
print("\nEvaluating model on test data...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Optional: Visualize some predictions
print("\nVisualizing some predictions...")
predictions = model.predict(X_test[:5])

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f"Pred: {np.argmax(predictions[i])}\nActual: {y_test[i]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('e:/current project/AIML summer training/Git project/Deep Learning project/predictions.png')
print("Predictions saved to predictions.png")
