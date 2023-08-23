import logging

from typing import Union
import numpy as np
import ast

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


def dump_image(conn: BlitzGateway,
               image: mm_schema.Image,
               target_dataset: DatasetWrapper,
               ):
    if isinstance(image, mm_schema.Image5D):
        # TZYXC -> zctyx
        image_data = np.array(image.data).reshape(
            (
                int(image.t.values[0]),
                int(image.z.values[0]),
                int(image.y.values[0]),
                int(image.x.values[0]),
                int(image.c.values[0])
            )
        ).transpose((1, 4, 0, 2, 3))
    elif isinstance(image, mm_schema.Image2D):
        image_data = np.array(image.data).reshape(
            (
                1,
                1,
                int(image.y.values[0]),
                int(image.x.values[0]),
                1
            )
        )
    elif isinstance(image, mm_schema.ImageAsNumpy):
        image_data = image.data
    else:
        module_logger.error(f"Unsupported image type for {image.name}: {image.class_name}")
        return None
    source_image_id = None
    try:
        source_image_id = omero_tools.get_object_from_url(image.source_image_url[0])[1]
    except IndexError:
        module_logger.info(f"No source image id provided for {image.name}")
    except ValueError:
        module_logger.info(f"Invalid source image url provided for {image.name}")
    # TODO: add channel labels to the output image
    return omero_tools.create_image_from_numpy_array(conn=conn,
                                                     data=image_data,
                                                     image_name=image.name,
                                                     image_description=f"{image.description}.\nSource image:{source_image_id}",
                                                     channel_labels=None,
                                                     dataset=target_dataset,
                                                     source_image_id=source_image_id,
                                                     channels_list=None,
                                                     force_whole_planes=False
                                                     )

    # TODO: We should consider that we might want to add metadata to an output image


def dump_roi(conn: BlitzGateway,
             roi: mm_schema.ROI,
             image: ImageWrapper,
             ):
    shapes = [SHAPE_TO_FUNCTION[type(shape)](shape) for shape in roi.shapes]

    return omero_tools.create_roi(
        conn=conn,
        image=image,
        shapes=shapes,
        name=roi.label,
        description=roi.description,
    )


def dump_tag(conn: BlitzGateway,
             tag: mm_schema.Tag,
             omero_object: Union[ImageWrapper, DatasetWrapper, ProjectWrapper],
             ):
    # TODO: if tag has an id, we should use that id
    return omero_tools.create_tag(
        conn=conn,
        tag_text=tag.text,
        tag_description=tag.description,
        omero_object=omero_object,
    )


def dump_key_value(conn: BlitzGateway,
                   key_values: mm_schema.KeyValues,
                   omero_object: Union[ImageWrapper, DatasetWrapper, ProjectWrapper],
                   ):
    return omero_tools.create_key_value(
        conn=conn,
        annotation=key_values._as_dict,
        omero_object=omero_object,
        annotation_name=key_values.class_name,
        annotation_description=key_values.class_class_uri,
        namespace=key_values.class_model_uri,
    )

def _eval(s):
    try:
        ev = ast.literal_eval(s)
        return ev
    except ValueError:
        corrected = "\'" + s + "\'"
        ev = ast.literal_eval(corrected)
        return ev

def _eval_types(table: mm_schema.TableAsDict):
    for column in table.columns.values():
        breakpoint()
        column.values = [_eval(v) for v in column.values]
    return table


def dump_table(conn: BlitzGateway,
               table: mm_schema.Table,
               omero_object: Union[ImageWrapper, DatasetWrapper, ProjectWrapper],
               ):
    if isinstance(table, mm_schema.TableAsDict):
        # linkML if casting everything as a string and we have to evaluate it back
        columns = {c.name: [_eval(v) for v in c.values] for c in table.columns.values()}
        return omero_tools.create_table(
            conn=conn,
            table=columns,
            table_name=table.name,
            omero_object=omero_object,
            table_description=table.description,
            namespace=table.class_model_uri,
        )
    elif isinstance(table, mm_schema.TableAsPandasDF):
        return omero_tools.create_table(
            conn=conn,
            table=table.df,
            table_name=table.name,
            omero_object=omero_object,
            table_description=table.description,
            namespace=table.class_model_uri,
        )
    else:
        module_logger.error(f"Unsupported table type for {table.name}: {table.class_name}")
        return None

def dump_comment(conn: BlitzGateway,
                 comment: mm_schema.Comment,
                 omero_object: Union[ImageWrapper, DatasetWrapper, ProjectWrapper],
                 ):
    return omero_tools.create_comment(
        conn=conn,
        comment_text=comment.text,
        omero_object=omero_object,
        namespace=comment.class_model_uri,
    )

