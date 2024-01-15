import math
import os
from typing import List, Tuple
from xml.etree import ElementTree

import numpy as np
from keras.src.utils import array_to_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.data_preparation.extract_grape_info import extract_grape_info


def main():
    should_normalise_colours: bool = True
    learn(should_normalise_colours)


def learn(should_normalise_colours: bool):
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
    train_items, test_items = train_test_split(data_items, test_size=0.3, random_state=42)
    validation_items, test_items = train_test_split(test_items, test_size=0.5, random_state=42)

    train_images: Tuple[np.ndarray, ...]
    train_bbch_values: Tuple[int, ...]
    validation_images: Tuple[np.ndarray, ...]
    validation_bbch_values: Tuple[int, ...]
    test_images: Tuple[np.ndarray, ...]
    test_bbch_values: Tuple[int, ...]

    train_images, train_bbch_values = zip(*[(item.image, item.bbch) for item in train_items])
    validation_images, validation_bbch_values = zip(*[(item.image, item.bbch) for item in validation_items])
    test_images, test_bbch_values = zip(*[(item.image, item.bbch) for item in test_items])


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


if __name__ == '__main__':
    main()
