import os
import xml.etree.ElementTree as ElementTree


def split_photos():
    # path to the resources directory
    resources_path = "../../resources"

    # get the list of all the files in the annotations folder
    annotations = os.listdir(resources_path + "/annotations")
    # get the list of all the files in the photos folder
    photos = os.listdir(resources_path + "/photos")

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
                # get the output folder path
                output_folder = resources_path + "/splitPhotos/" + file_name
                # if the output folder does not exist
                if not os.path.exists(output_folder):
                    # create the output folder
                    os.makedirs(output_folder)
                # split the photo
                split_photo(photo_path, annotation_path, output_folder)


def split_photo(photo_path: str, annotation_path: str, output_folder: str):
    # parse the XML file
    tree = ElementTree.parse(annotation_path)
    root = tree.getroot()

    # for each 'object' in the annotation
    for obj in root.findall('object'):
        # get the bounding box
        bndbox = obj.find('bndbox')
        # get the bounding box values
        x_min = int(bndbox.find('xmin').text)
        y_min = int(bndbox.find('ymin').text)
        x_max = int(bndbox.find('xmax').text)
        y_max = int(bndbox.find('ymax').text)

        # get the width of the bounding box
        width = x_max - x_min
        # get the height of the bounding box
        height = y_max - y_min

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
    photo = open(photo_path, "rb")
    # read the photo
    photo_read = photo.read()
    # close the photo
    photo.close()

    # crop the photo
    cropped_photo = photo_read[y_min:y_min + height, x_min:x_min + width]

    # open the output file
    output_file = open(output_file_path, "wb")
    # write the cropped photo to the output file
    output_file.write(cropped_photo)
    # close the output file
    output_file.close()


if __name__ == "__main__":
    split_photos()
