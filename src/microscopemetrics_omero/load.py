import logging
from datetime import datetime


from microscopemetrics_omero import omero_tools
from omero.gateway import BlitzGateway, ImageWrapper
import numpy as np

from dump import dump_image_process

# Creating logging services
module_logger = logging.getLogger("microscopemetrics_omero.load")


def load_image(image: ImageWrapper) -> np.ndarray:
    return omero_tools.get_image_intensities(image)


