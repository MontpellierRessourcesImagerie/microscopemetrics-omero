import logging

from typing import Union
import numpy as np

from dataclasses import fields

from microscopemetrics_omero import omero_tools
from microscopemetrics.data_schema import core_schema as mm_schema
from omero.gateway import BlitzGateway, ImageWrapper, DatasetWrapper, ProjectWrapper

# Creating logging services
module_logger = logging.getLogger("microscopemetrics_omero.dump")

SHAPE_TO_FUNCTION = {
    mm_schema.Point: omero_tools.create_shape_point,
    mm_schema.Line: omero_tools.create_shape_line,
    mm_schema.Rectangle: omero_tools.create_shape_rectangle,
    mm_schema.Ellipse: omero_tools.create_shape_ellipse,
    mm_schema.Polygon: omero_tools.create_shape_polygon,
    mm_schema.Mask: omero_tools.create_shape_mask,
}


def dump_output_image(conn: BlitzGateway,
                      output_image: mm_schema.Image,
                      target_dataset: DatasetWrapper,
                      ):
    if isinstance(output_image, mm_schema.Image5D):
        # TZYXC -> zctyx
        image_data = np.array(output_image.data).reshape(
            (
                int(output_image.t.values[0]),
                int(output_image.z.values[0]),
                int(output_image.y.values[0]),
                int(output_image.x.values[0]),
                int(output_image.c.values[0])
            )
        ).transpose((1, 4, 0, 2, 3))
    elif isinstance(output_image, mm_schema.Image2D):
        image_data = np.array(output_image.data).reshape(
            (
                1,
                1,
                int(output_image.y.values[0]),
                int(output_image.x.values[0]),
                1
            )
        )
    elif isinstance(output_image, mm_schema.ImageAsNumpy):
        image_data = output_image.data
    else:
        module_logger.error(f"Unsupported image type for {output_image.name}: {output_image.class_name}")
        return None
    source_image_id = None
    try:
        source_image_id = omero_tools.get_object_from_url(output_image.source_image_url[0])[1]
    except IndexError:
        module_logger.info(f"No source image id provided for {output_image.name}")
    except ValueError:
        module_logger.info(f"Invalid source image url provided for {output_image.name}")
    # TODO: add channel labels to the output image
    return omero_tools.create_image_from_numpy_array(conn=conn,
                                                     data=image_data,
                                                     image_name=output_image.name,
                                                     image_description=f"{output_image.description}.\nSource image:{source_image_id}",
                                                     channel_labels=None,
                                                     dataset=target_dataset,
                                                     source_image_id=source_image_id,
                                                     channels_list=None,
                                                     force_whole_planes=False
                                                     )

    # TODO: We should consider that we might want to add metadata to an output image


def dump_output_roi(conn: BlitzGateway,
                    output_roi: mm_schema.ROI,
                    image: ImageWrapper,
                    ):
    shapes = [SHAPE_TO_FUNCTION[type(shape)](shape) for shape in output_roi.shapes]

    return omero_tools.create_roi(
        conn=conn,
        image=image,
        shapes=shapes,
        name=output_roi.label,
        description=output_roi.description,
    )


def dump_output_tag(conn: BlitzGateway,
                    output_tag: mm_schema.Tag,
                    omero_object: Union[ImageWrapper, DatasetWrapper, ProjectWrapper],
                    ):
    # TODO: if tag has an id, we should use that id
    return omero_tools.create_tag(
        conn=conn,
        tag=output_tag.text,
        omero_object=omero_object,
    )


def dump_output_key_value(conn: BlitzGateway,
                          output_key_values: mm_schema.KeyValues,
                          omero_object: Union[ImageWrapper, DatasetWrapper, ProjectWrapper],
                          ):
    return omero_tools.create_key_value(
        conn=conn,
        annotation=output_key_values.key_values,
        omero_object=omero_object,
        annotation_name=output_key_values.name,
        annotation_description=output_key_values.description,
        namespace=output_key_values.class_model_uri,
    )


def dump_output_table(conn: BlitzGateway,
                      output_table: mm_schema.Table,
                      omero_object: Union[ImageWrapper, DatasetWrapper, ProjectWrapper],
                      ):
    return omero_tools.create_table(
        conn=conn,
        table=output_table.table,
        table_name=output_table.name,
        omero_object=omero_object,
        table_description=output_table.description,
        namespace=output_table.class_model_uri,
    )


def dump_comment(conn: BlitzGateway,
                 output_comment: mm_schema.Comment,
                 omero_object: Union[ImageWrapper, DatasetWrapper, ProjectWrapper],
                 ):
    return omero_tools.create_comment(
        conn=conn,
        comment_value=output_comment.comment,
        omero_object=omero_object,
        namespace=output_comment.class_model_uri,
    )

