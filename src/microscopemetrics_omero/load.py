import logging
from datetime import datetime


import omero_tools as omero_tools
from microscopemetrics import samples
from omero.gateway import BlitzGateway, ImageWrapper
import numpy as np

from dump import dump_image_process

# Creating logging services
module_logger = logging.getLogger("microscopemetrics_omero.load")


def load_image(image: ImageWrapper) -> np.ndarray:
    return omero_tools.get_image_intensities(image)


