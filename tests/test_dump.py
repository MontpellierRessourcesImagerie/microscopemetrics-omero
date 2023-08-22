from numpy import ndarray
import microscopemetrics_omero.dump as dump
from omero.gateway import ImageWrapper, DatasetWrapper, ProjectWrapper, RoiWrapper

# def test_dump_image_process(conn, mm_finished_analysis, project_structure):
#     image_info = project_structure[2]
#     im_id = image_info[0][1]
#     image = conn.getObject("Image", im_id)
#     dump.dump_image_process(conn, image, mm_finished_analysis, "test_namespace")
#
#
def test_dump_image(conn, mm_image5d_fixture, project_structure):
    image_info = project_structure[2]
    im_id = image_info[0][1]
    image = conn.getObject("Image", im_id)
    target = image.getParent()

    image_obj = dump.dump_output_image(conn=conn,
                                       output_image=mm_image5d_fixture,
                                       target_dataset=target
                                       )
    assert type(image_obj) == ImageWrapper
    assert type(image_obj.getId()) == int


def test_dump_roi(conn, mm_roi_fixture, project_structure):
    image_info = project_structure[2]
    im_id = image_info[0][1]
    image = conn.getObject("Image", im_id)
    roi_obj = dump.dump_output_roi(conn=conn,
                                   output_roi=mm_roi_fixture,
                                   image=image)
    breakpoint()

def test_dump_tag(conn, mm_tag_fixture, project_structure):
    image_info = project_structure[2]
    im_id = image_info[0][1]
    image = conn.getObject("Image", im_id)
    # TODO: test image, dataset and project
    pass


# def test_dump_output_key_value(conn, mm_key_values_fixture, project_structure):
#     image_info = project_structure[2]
#     im_id = image_info[0][1]
#     image = conn.getObject("Image", im_id)
#     # TODO: test image, dataset and project
#     pass
#
#
# def test_dump_output_table(conn, mm_table_fixture, project_structure):
#     image_info = project_structure[2]
#     im_id = image_info[0][1]
#     image = conn.getObject("Image", im_id)
#     # TODO: test image, dataset and project
#     pass
#
#
# def test_dump_comment(conn, mm_comment_fixture, project_structure):
#     image_info = project_structure[2]
#     im_id = image_info[0][1]
#     image = conn.getObject("Image", im_id)
#     # TODO: test image, dataset and project
#     pass
#

