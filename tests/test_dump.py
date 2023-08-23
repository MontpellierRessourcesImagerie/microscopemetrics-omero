from numpy import ndarray
import microscopemetrics_omero.dump as dump
from omero.gateway import (
    ImageWrapper,
    DatasetWrapper,
    ProjectWrapper,
    RoiWrapper,
    TagAnnotationWrapper,
    MapAnnotationWrapper,
    FileAnnotationWrapper,
    CommentAnnotationWrapper,
)

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

    image_obj = dump.dump_image(conn=conn,
                                image=mm_image5d_fixture,
                                target_dataset=target
                                )
    assert image_obj is not None
    assert type(image_obj) == ImageWrapper
    assert type(image_obj.getId()) == int


def test_dump_roi(conn, mm_roi_fixture, project_structure):
    image_info = project_structure[2]
    im_id = image_info[0][1]
    image = conn.getObject("Image", im_id)
    roi_obj = dump.dump_roi(conn=conn,
                            roi=mm_roi_fixture,
                            image=image)

    assert roi_obj is not None
    assert type(roi_obj) == RoiWrapper
    assert type(roi_obj.getId()) == int


def test_dump_tag(conn, mm_tag_fixture, project_structure):
    image_info = project_structure[2]
    im_id = image_info[0][1]
    image = conn.getObject("Image", im_id)
    dataset = image.getParent()
    project = dataset.getParent()

    # this is creating separate tags
    image_tag_obj = dump.dump_tag(conn=conn,
                                  tag=mm_tag_fixture,
                                  omero_object=image)
    dataset_tag_obj = dump.dump_tag(conn=conn,
                                    tag=mm_tag_fixture,
                                    omero_object=dataset)
    project_tag_obj = dump.dump_tag(conn=conn,
                                    tag=mm_tag_fixture,
                                    omero_object=project)

    assert image_tag_obj is not None
    assert type(image_tag_obj) == TagAnnotationWrapper
    assert type(image_tag_obj.getId()) == int
    assert dataset_tag_obj is not None
    assert type(dataset_tag_obj) == TagAnnotationWrapper
    assert type(dataset_tag_obj.getId()) == int
    assert project_tag_obj is not None
    assert type(project_tag_obj) == TagAnnotationWrapper
    assert type(project_tag_obj.getId()) == int


def test_dump_key_value(conn, mm_key_values_fixture, project_structure):
    image_info = project_structure[2]
    im_id = image_info[0][1]
    image = conn.getObject("Image", im_id)
    dataset = image.getParent()
    project = dataset.getParent()

    image_kv_obj = dump.dump_key_value(conn=conn,
                                       key_values=mm_key_values_fixture,
                                       omero_object=image)
    dataset_kv_obj = dump.dump_key_value(conn=conn,
                                         key_values=mm_key_values_fixture,
                                         omero_object=dataset)
    project_kv_obj = dump.dump_key_value(conn=conn,
                                         key_values=mm_key_values_fixture,
                                         omero_object=project)

    assert image_kv_obj is not None
    assert type(image_kv_obj) == MapAnnotationWrapper
    assert type(image_kv_obj.getId()) == int
    assert dataset_kv_obj is not None
    assert type(dataset_kv_obj) == MapAnnotationWrapper
    assert type(dataset_kv_obj.getId()) == int
    assert project_kv_obj is not None
    assert type(project_kv_obj) == MapAnnotationWrapper
    assert type(project_kv_obj.getId()) == int


def test_dump_table(conn, mm_table_as_dict_fixture, project_structure):
    image_info = project_structure[2]
    im_id = image_info[0][1]
    image = conn.getObject("Image", im_id)
    dataset = image.getParent()
    project = dataset.getParent()

    image_table_obj = dump.dump_table(conn=conn,
                                      table=mm_table_as_dict_fixture,
                                      omero_object=image)
    dataset_table_obj = dump.dump_table(conn=conn,
                                        table=mm_table_as_dict_fixture,
                                        omero_object=dataset)
    project_table_obj = dump.dump_table(conn=conn,
                                        table=mm_table_as_dict_fixture,
                                        omero_object=project)

    assert image_table_obj is not None
    assert type(image_table_obj) == FileAnnotationWrapper
    assert type(image_table_obj.getId()) == int
    assert dataset_table_obj is not None
    assert type(dataset_table_obj) == FileAnnotationWrapper
    assert type(dataset_table_obj.getId()) == int
    assert project_table_obj is not None
    assert type(project_table_obj) == FileAnnotationWrapper
    assert type(project_table_obj.getId()) == int


def test_dump_comment(conn, mm_comment_fixture, project_structure):
    image_info = project_structure[2]
    im_id = image_info[0][1]
    image = conn.getObject("Image", im_id)
    dataset = image.getParent()
    project = dataset.getParent()

    image_comment_obj = dump.dump_comment(conn=conn,
                                          comment=mm_comment_fixture,
                                          omero_object=image)
    dataset_comment_obj = dump.dump_comment(conn=conn,
                                            comment=mm_comment_fixture,
                                            omero_object=dataset)
    project_comment_obj = dump.dump_comment(conn=conn,
                                            comment=mm_comment_fixture,
                                            omero_object=project)

    assert image_comment_obj is not None
    assert type(image_comment_obj) == CommentAnnotationWrapper
    assert type(image_comment_obj.getId()) == int
    assert dataset_comment_obj is not None
    assert type(dataset_comment_obj) == CommentAnnotationWrapper
    assert type(dataset_comment_obj.getId()) == int
    assert project_comment_obj is not None
    assert type(project_comment_obj) == CommentAnnotationWrapper
    assert type(project_comment_obj.getId()) == int



