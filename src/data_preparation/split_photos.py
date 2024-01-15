import os
import xml.etree.ElementTree as ElementTree
from PIL import Image

from src.data_preparation.extract_grape_info import extract_grape_info


def split_photos():
    # path to the resources directory
    resources_path = "../../resources"

    # get the list of all the files in the annotations folder
    annotations = os.listdir(resources_path + "/annotations")
    # get the list of all the files in the photos folder
    photos = os.listdir(resources_path + "/photos")

    output_folder = resources_path + "/splitPhotos"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # remove all .jpg files in the output directory
    for file in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file)
        try:
            if os.path.isfile(file_path) and file.endswith(".jpg"):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    # for each file in the annotations folder
    for annotation in annotations:
        # if the file is a .xml file
        if annotation.endswith(".xml"):
            # get the file name without the extension
            file_name = annotation[:-4]
            # get the file name with the extension
            file_name_with_extension = annotation
            # get the corresponding photo file name
            photo_name = file_name + ".jpg"
            # if the photo file exists
            if photo_name in photos:
                # get the photo file path
                photo_path = resources_path + "/photos/" + photo_name
                # get the annotation file path
                annotation_path = resources_path + "/annotations/" + file_name_with_extension
                # split the photo
                split_photo(photo_path, annotation_path, output_folder)


def split_photo(photo_path: str, annotation_path: str, output_folder: str):
    # parse the XML file
    tree = ElementTree.parse(annotation_path)
    root = tree.getroot()

    # for each 'object' in the annotation
    for obj in root.findall('object'):
        x_min, y_min, width, height, _ = extract_grape_info(obj)

        # get the photo file name
        photo_name = photo_path[photo_path.rfind("/") + 1:]
        # get the output file name
        output_file_name = photo_name[:-4] + "_" + str(x_min) + "_" + str(y_min) + "_" + str(width) + "_" + str(
            height) + ".jpg"
        # get the output file path
        output_file_path = output_folder + "/" + output_file_name
        # crop the photo
        crop_photo(photo_path, output_file_path, x_min, y_min, width, height)


def crop_photo(photo_path: str, output_file_path: str, x_min: int, y_min: int, width: int, height: int):
    # open the photo
    photo = Image.open(photo_path)

    # crop the photo
    cropped_photo = photo.crop((x_min, y_min, x_min + width, y_min + height))

    # save the cropped photo to the output file
    cropped_photo.save(output_file_path)


if __name__ == "__main__":
    split_photos()
