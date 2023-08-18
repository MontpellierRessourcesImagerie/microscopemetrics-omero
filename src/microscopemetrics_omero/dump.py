import logging

from microscopemetrics_omero import omero_tools
from microscopemetrics import samples as mm_samples
from microscopemetrics.data_schema import core_schema as mm_schema
from omero.gateway import BlitzGateway, ImageWrapper

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

def dump_image_process(
    conn: BlitzGateway,
    image: ImageWrapper,
    analysis: mm_samples.Analysis,
    namespace: str
) -> None:
    property_types_to_dump = {
        "Image": _dump_output_image,
        "Roi": _dump_output_roi,
        "Tag": _dump_output_tag,
        "KeyValues": _dump_output_key_value,
        "Table": _dump_output_table,
        "Comment": _dump_comment,
    }
    for property_name, output_property in analysis.output.properties.items():
        # TODO: add a try/except to catch errors and log them
        module_logger.info(
            f"Dumping {output_property.type} {property_name} from analysis {analysis.get_name()}"
        )
        property_types_to_dump[output_property.type](conn, output_property, image, namespace)
        module_logger.info(
            f"Dumping {output_property.type} {output_property.name} from analysis {analysis.get_name()} completed"
        )


def _dump_output_image(conn: BlitzGateway, output_image: mm_schema.Image, source_image, namespace):
    # TODO: add channel labels to the output image
    omero_tools.create_image_from_numpy_array(conn=conn,
                                              data=output_image.data,
                                              image_name=output_image.name,
                                              image_description=f"{output_image.description}.\n" f"Source image:{source_image.getId()}",
                                              channel_labels=None,
                                              dataset=source_image.getParent(),
                                              source_image_id=source_image.getId(),
                                              channels_list=None,
                                              force_whole_planes=False
                                              )

    # TODO: We should consider that we might want to add metadata to an output image


def _dump_output_roi(conn, output_roi, image, namespace):
    shapes = [SHAPE_TO_FUNCTION[type(shape)](shape) for shape in output_roi.shapes]
    omero_tools.create_roi(
        conn=conn,
        image=image,
        shapes=shapes,
        name=output_roi.name,
        description=output_roi.description,
    )


def _dump_output_tag(conn: BlitzGateway, output_tag: mm, omero_object, namespace):
    omero_tools.create_tag(
        conn=conn,
        tag=output_tag.tag_value,
        omero_object=omero_object,
    )


def _dump_output_key_value(conn, output_key_values, omero_object, namespace):
    omero_tools.create_key_value(
        conn=conn,
        annotation=output_key_values.key_values,
        omero_object=omero_object,
        annotation_name=output_key_values.name,
        annotation_description=output_key_values.description,
        namespace=namespace,
    )


def _dump_output_table(conn, output_table, omero_object, namespace):
    omero_tools.create_table(
        conn=conn,
        table=output_table.table,
        table_name=output_table.name,
        omero_object=omero_object,
        table_description=output_table.description,
        namespace=namespace,
    )


def _dump_comment(conn, output_comment, omero_object, namespace):
    omero_tools.create_comment(
        conn=conn,
        comment_value=output_comment.comment,
        omero_object=omero_object,
        namespace=namespace,
    )

