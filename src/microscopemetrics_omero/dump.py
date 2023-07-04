import logging

import omero_tools as omero_tools
from microscopemetrics import samples
from omero.gateway import BlitzGateway, ImageWrapper

# Creating logging services
module_logger = logging.getLogger("microscopemetrics_omero.dump")


def dump_image_process(
    conn: BlitzGateway,
    image: ImageWrapper,
    analysis: samples.Analysis,
) -> None:
    property_types_to_dump = {
        "Image": _dump_output_image,
        "Roi": _dump_output_roi,
        "Tag": _dump_output_tag,
        "KeyValues": _dump_output_key_value,
        "Table": _dump_output_table,
        "Comment": _dump_comment,
    }
    for output_property in analysis.output.get_properties():
        # TODO: add a try/except to catch errors and log them
        module_logger.info(
            f"Dumping {output_property.type} {output_property.name} from analysis {analysis.name}"
        )
        property_types_to_dump[output_property.type](conn, output_property, image)
        module_logger.info(
            f"Dumping {output_property.type} {output_property.name} from analysis {analysis.name} completed"
        )


def _dump_output_image(conn, output_image, image):
    new_image = omero_tools.create_image(
        conn=conn,
        image=output_image.data,
        source_image=image,
        image_name=output_image.name,
        description=f"{output_image.description}.\n" f"Source image_id:{image.getId()}",
    )

    # TODO: add metadata to image


def _dump_output_roi(conn, output_roi, image):
    omero_tools.create_roi(
        conn=conn,
        image=image,
        shapes=output_roi.shapes,
        name=output_roi.name,
        description=output_roi.description,
    )


def _dump_output_tag(conn, output_tag, object):
    omero_tools.create_tag(
        conn=conn,
        tag_string=output_tag.tag_value,
        omero_object=object,
        description=output_tag.description,
    )


def _dump_output_key_value(conn, output_key_values, omero_object):
    omero_tools.create_key_value(
        conn=conn,
        annotation=output_key_values.key_values,
        omero_object=omero_object,
        annotation_name=output_key_values.name,
        annotation_description=output_key_values.description,
        namespace=generate_namespace(),
    )


def _dump_output_table(conn, output_table, omero_object):
    omero_tools.create_table(
        conn=conn,
        table=output_table.table,
        table_name=output_table.name,
        omero_object=omero_object,
        table_description=output_table.description,
        namespace=generate_namespace(),
    )


def _dump_comment(conn, output_comment, omero_object):
    omero_tools.create_comment(
        conn=conn,
        comment_value=output_comment.comment,
        omero_object=omero_object,
        namespace=generate_namespace(),
    )

