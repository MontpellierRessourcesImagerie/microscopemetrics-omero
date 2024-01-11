from numpy import ndarray

import microscopemetrics_omero.load as load


def test_load_image(conn, numpy_image_fixture, project_structure):
    image_info = project_structure["image_info"]
    im_id = image_info["image_0.czi"]
    image = conn.getObject("Image", im_id)
    data = load.load_image(image)
    assert type(data) == ndarray
