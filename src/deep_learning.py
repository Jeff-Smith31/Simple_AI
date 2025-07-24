import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Enable GPU memory growth if you're using a GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)



class DeepLearningExample:
    def __init__(self):
        # Load and preprocess the MNIST dataset
        self.load_data()
        # Create the model architecture
        self.create_model()

    def load_data(self):
        print("Loading and preprocessing data...")
        # Load MNIST dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

        # Normalize pixel values to be between 0 and 1
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0

        # Reshape images to specify they're grayscale
        self.x_train = self.x_train.reshape((-1, 28, 28, 1))
        self.x_test = self.x_test.reshape((-1, 28, 28, 1))

        # Convert class vectors to binary class matrices (one-hot encoding)
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, 10)

        print(f"Training samples: {self.x_train.shape[0]}")
        print(f"Test samples: {self.x_test.shape[0]}")

    def create_model(self):
        print("\nCreating model architecture...")
        self.model = tf.keras.Sequential([
            # Convolutional layers
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

            # Flatten the 2D feature maps for the fully connected layers
            tf.keras.layers.Flatten(),

            # Fully connected layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),  # Dropout layer to prevent overfitting
            tf.keras.layers.Dense(10, activation='softmax')  # Output layer (10 digits)
        ])

        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Print model summary
        self.model.summary()

    def train_model(self, epochs=5, batch_size=64):
        print("\nTraining the model...")

        # Create a timestamp for the log directory
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        # Callbacks for training
        callbacks = [
            # TensorBoard callback for visualization
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
            # Early stopping to prevent overfitting
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            # Model checkpoint to save the best model
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]

        # Train the model
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks
        )

    def evaluate_model(self):
        print("\nEvaluating the model...")
        # Evaluate the model on the test set
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")

    def plot_training_history(self):
        print("\nPlotting training history...")
        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def make_predictions(self, num_samples=5):
        print("\nMaking predictions on sample images...")
        # Get random test samples
        indices = np.random.randint(0, len(self.x_test), num_samples)
        sample_images = self.x_test[indices]
        true_labels = np.argmax(self.y_test[indices], axis=1)

        # Make predictions
        predictions = self.model.predict(sample_images)
        predicted_labels = np.argmax(predictions, axis=1)

        # Plot the results
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        for i, (image, true_label, pred_label) in enumerate(
                zip(sample_images, true_labels, predicted_labels)
        ):
            axes[i].imshow(image.reshape(28, 28), cmap='gray')
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()


def main():
    # Create an instance of our deep learning example
    dl_example = DeepLearningExample()

    # Train the model
    dl_example.train_model(epochs=10)

    # Evaluate the model
    dl_example.evaluate_model()

    # Plot training history
    dl_example.plot_training_history()

    # Make and visualize predictions
    dl_example.make_predictions()


if __name__ == "__main__":
    main()
