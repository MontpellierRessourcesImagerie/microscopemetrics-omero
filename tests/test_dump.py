from numpy import ndarray
import microscopemetrics_omero.dump as dump

def test_dump_image_process(conn, microscopemetrics_finished_analysis, project_structure):
    image_info = project_structure[2]
    im_id = image_info[0][1]
    image = conn.getObject("Image", im_id)
    dump.dump_image_process(conn, image, microscopemetrics_finished_analysis, "test_namespace")

#
# def test_dump_output_image(conn, microscopemetrics_finished_analysis, project_structure):
#     image_info = project_structure[2]
#     im_id = image_info[0][1]
#     image = conn.getObject("Image", im_id)
#     pass
#
#
# def test_dump_output_roi(conn, mm_roi_fixture, project_structure):
#     image_info = project_structure[2]
#     im_id = image_info[0][1]
#     image = conn.getObject("Image", im_id)
#     pass
#
#
# def test_dump_output_tag(conn, mm_tag_fixture, project_structure):
#     image_info = project_structure[2]
#     im_id = image_info[0][1]
#     image = conn.getObject("Image", im_id)
#     # TODO: test image, dataset and project
#     pass
#
#
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

