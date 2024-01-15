# use amd gpu
from os import environ
environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras

from enum import Enum

from keras.optimizers import Adam

import time
import os
from typing import List, Tuple
from xml.etree import ElementTree

import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import array_to_img
from keras.callbacks import Callback, History, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array

from src.data_preparation.extract_grape_info import extract_grape_info

import matplotlib.pyplot as plt


class ModelSize(Enum):
    NORMAL = 1
    BIGGER = 2
    BIGGEST = 3


def main():
    # Define the hyperparameters and options to test
    should_normalise_colours_options: List[bool] = [True, False]
    model_sizes: List[ModelSize] = [ModelSize.NORMAL, ModelSize.BIGGER, ModelSize.BIGGEST]
    learning_rates = [0.0005, 0.001, 0.002, 0.005]

    results_dir = "../../results"

    # Loop over all combinations of hyperparameters and options
    for should_normalise_colours in should_normalise_colours_options:
        for model_size in model_sizes:
            for lr in learning_rates:
                print(
                    f"\nTraining model with color normalization: {should_normalise_colours}, model size: {model_size}, learning rate: {lr}")
                test_loss, test_accuracy, history = learn(should_normalise_colours, model_size, lr)

                # Write the results to a file
                results_file = os.path.join(results_dir, f"results_{should_normalise_colours}_{model_size}_{lr}.txt")
                with open(results_file, "w") as f:
                    f.write(f"Test Loss: {test_loss}\n")
                    f.write(f"Test Accuracy: {test_accuracy}\n")

                # Create a plot of the accuracy over the epochs
                plt.figure()
                epochs = list(range(1, len(history.history['acc']) + 1))
                plt.plot(epochs, history.history['acc'])
                plt.plot(epochs, history.history['val_acc'])
                plt.title('Model accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper left')

                # Save the plot to the results directory
                plot_file = os.path.join(results_dir, f"plot_{should_normalise_colours}_{model_size}_{lr}.png")
                plt.savefig(plot_file)


def learn(should_normalise_colours: bool, model_size: ModelSize, learning_rate: float) -> Tuple[float, float, History]:
    # Load and preprocess the data
    resources_dir: str = '../../resources/'
    image_dir: str = resources_dir + 'splitPhotos'
    image_files: List[str] = os.listdir(image_dir)
    images: dict = {file[:-4]: img_to_array(load_img(os.path.join(image_dir, file))) for file in image_files if
                    file.endswith('.jpg')}

    annotation_dir: str = resources_dir + 'annotations'
    annotation_files: List[str] = os.listdir(annotation_dir)
    annotations: dict = {file[:-4]: load_annotations(os.path.join(annotation_dir, file)) for file in annotation_files}

    # Pair each image with its annotation
    paired_data: dict = {}
    for filename, image in images.items():
        uuid, x_min, y_min, width, height = filename.split('_')
        for annotation in annotations[uuid]:
            if annotation[:4] == (int(x_min), int(y_min), int(width), int(height)):
                paired_data[filename] = ImageAnnotationPair(filename, image, annotation)
                break

    if should_normalise_colours:
        # Normalize the images (scale all pixel values to be between 0 and 1)
        for filename, image_annotation_pair in paired_data.items():
            image: np.ndarray = image_annotation_pair.image
            annotation: Tuple[int, int, int, int, int] = (image_annotation_pair.x_min, image_annotation_pair.y_min,
                                                          image_annotation_pair.width, image_annotation_pair.height,
                                                          image_annotation_pair.bbch)

            paired_data[filename] = ImageAnnotationPair(filename, np.array(image) / 255.0, annotation)

    # Resize each image in the paired_data dictionary
    for filename, image_annotation_pair in paired_data.items():
        image: np.ndarray = image_annotation_pair.image
        annotation: Tuple[int, int, int, int, int] = (image_annotation_pair.x_min, image_annotation_pair.y_min,
                                                      image_annotation_pair.width, image_annotation_pair.height,
                                                      image_annotation_pair.bbch)

        resized_image: np.ndarray = resize_image(image, (128, 128))  # Resize to 128x128
        paired_data[filename] = ImageAnnotationPair(filename, resized_image, annotation)

    train_items: List[ImageAnnotationPair]
    test_items: List[ImageAnnotationPair]
    validation_items: List[ImageAnnotationPair]

    # Split the data into training, validation, and test sets
    # Training set: 70%
    # Validation set: 15% (30% * 0.5)
    # Test set: 15% (30% * 0.5)
    data_items: List[ImageAnnotationPair] = list(paired_data.values())
    train_items, test_items = train_test_split(data_items, test_size=0.3, random_state=42, shuffle=True)
    validation_items, test_items = train_test_split(test_items, test_size=0.5, random_state=42, shuffle=True)

    train_images, train_bbch_values = zip(*[(item.image, item.bbch) for item in train_items])
    validation_images, validation_bbch_values = zip(*[(item.image, item.bbch) for item in validation_items])
    test_images, test_bbch_values = zip(*[(item.image, item.bbch) for item in test_items])

    model = Sequential()

    bbch_to_class = {71: 0, 73: 1, 75: 2, 77: 3, 79: 4}
    class_to_bbch = {v: k for k, v in bbch_to_class.items()}

    train_bbch_values = [bbch_to_class[bbch] for bbch in train_bbch_values]
    validation_bbch_values = [bbch_to_class[bbch] for bbch in validation_bbch_values]
    test_bbch_values = [bbch_to_class[bbch] for bbch in test_bbch_values]

    # One-hot encode the BBCH values (convert them to categorical values, such as 71 -> [1, 0, 0, 0, 0])
    train_bbch_values = to_categorical(train_bbch_values, num_classes=len(bbch_to_class))
    validation_bbch_values = to_categorical(validation_bbch_values, num_classes=len(bbch_to_class))
    test_bbch_values = to_categorical(test_bbch_values, num_classes=len(bbch_to_class))

    # Add layers to the model
    if model_size == ModelSize.NORMAL:
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
    elif model_size == ModelSize.BIGGER:
        # More filters in convolutional layers and more neurons in dense layers
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
    else:
        # this model adds more convolutional layers and utilises dropout layers in order to reduce data overfitting
        # First convolutional block
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Second convolutional block
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Third convolutional block
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

    model.add(Dense(len(bbch_to_class), activation='softmax'))

    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Evaluate the untrained model with the test data
    untrained_test_loss, untrained_test_accuracy = model.evaluate(np.array(test_images), np.array(test_bbch_values), verbose=2)

    # Add early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)

    # Fit the model with validation data
    time_callback = TimeHistory()
    history: History = model.fit(np.array(train_images), train_bbch_values,
              validation_data=(np.array(validation_images), validation_bbch_values),
              epochs=50, shuffle=True, callbacks=[time_callback, early_stopping])

    # After training, evaluate the model with the test data
    test_loss, test_accuracy = model.evaluate(np.array(test_images), np.array(test_bbch_values))
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Untrained model, Test Loss: {untrained_test_loss}")
    print(f"Untrained model, Test Accuracy: {untrained_test_accuracy}")

    return test_loss, test_accuracy, history


class ImageAnnotationPair:
    def __init__(self, filename: str, image: np.ndarray, annotation: Tuple[int, int, int, int, int]):
        self.filename: str = filename
        self.image: np.ndarray = image
        self.x_min: int = annotation[0]
        self.y_min: int = annotation[1]
        self.width: int = annotation[2]
        self.height: int = annotation[3]
        self.bbch: int = annotation[4]


def load_annotations(file_path: str) -> List[Tuple[int, int, int, int, int]]:
    # parse the XML file
    tree = ElementTree.parse(file_path)
    root = tree.getroot()

    annotations = [extract_grape_info(obj) for obj in root.findall('object')]
    return annotations


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return img_to_array(array_to_img(image, scale=False).resize(size))


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        time_taken = time.time() - self.epoch_time_start
        self.times.append(time_taken)
        print(f"\nTime taken for epoch {batch + 1}: {time_taken} seconds")


if __name__ == '__main__':
    main()
