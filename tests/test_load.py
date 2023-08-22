from numpy import ndarray
import microscopemetrics_omero.load as load

def test_load_image(conn, numpy_image_fixture, project_structure):
    image_info = project_structure[2]
    im_id = image_info[0][1]
    image = conn.getObject("Image", im_id)
    data = load.load_image(image)
    assert type(data) == ndarray
