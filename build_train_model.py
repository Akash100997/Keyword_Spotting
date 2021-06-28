import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

# Load the data

DATA_PATH = "data.npz"
TRAINED_MODEL_PATH = "model.h5"
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 40


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
    :param history: Training history of model
    :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()


def load_dataset(dataset_path):
    features = np.load(dataset_path)

    x = features["MFCCs"]
    y = features["labels"]

    return x, y


def dataset_split(dataset_path, test_size=0.2, validation_test_size=0.2):
    # Load the dataset
    x, y = load_dataset(dataset_path)

    # Split the train and test data's
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_size)

    # Split the train and Validation Data's
    X_train, X_Validation, Y_train, Y_Validation = train_test_split(X_train, Y_train, test_size=validation_test_size)

    # Convert the 2 Dimensional Array to 4D, so that we can build a CNN model
    X_train = X_train[..., np.newaxis]
    X_Validation = X_Validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Return the datasets
    return X_train, X_Validation, X_test, Y_train, Y_Validation, Y_test


def build_model(Input_shape):
    # Build network Architecture using Convolutional Layers
    model = keras.models.Sequential()

    # Convolution Layer 1
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=Input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # Convolution Layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # Convolution Layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    # Flatten the Layer and pass it to the Dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # Apply Softmax activation function
    model.add(keras.layers.Dense(10, activation='softmax'))

    # Specify the optimiser for compiling the model
    optimiser = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = keras.losses.SparseCategoricalCrossentropy()

    # Compile the Model
    model.compile(optimizer=optimiser, loss=loss, metrics=["accuracy"])

    # Print the Model Summary
    model.summary()

    # Return the Model
    return model


def main():
    # Split the train, validation and test dataset from the json file
    x_train, x_validation, x_test, y_train, y_validation, y_test = dataset_split(DATA_PATH)

    # Get the Input Shape
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

    # Build the model
    model = build_model(input_shape)

    # Train the model
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=(x_validation, y_validation))

    # plot accuracy/loss for training/validation set as a function of the epochs
    plot_history(history)

    # Evaluate the network on test set
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("\nTest loss: {}, Test accuracy: {}".format(test_loss, 100 * test_acc))

    # Save the Model
    model.save(TRAINED_MODEL_PATH)


if __name__ == "__main__":
    main()
