from typing import Tuple
from xml.etree import ElementTree


def extract_grape_info(obj: ElementTree.Element) -> Tuple[int, int, int, int, int]:
    # get the bounding box
    bndbox = obj.find('bndbox')
    # get the bounding box values
    x_min = int(float(bndbox.find('xmin').text))
    y_min = int(float(bndbox.find('ymin').text))
    x_max = int(float(bndbox.find('xmax').text))
    y_max = int(float(bndbox.find('ymax').text))

    # get the width and height of the bounding box
    width = x_max - x_min
    height = y_max - y_min

    # get the BBCH value
    bbch = int(obj.find('attributes/attribute[name="BBCH"]/value').text)

    return x_min, y_min, width, height, bbch
