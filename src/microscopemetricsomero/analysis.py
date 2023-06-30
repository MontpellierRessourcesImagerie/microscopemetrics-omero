import logging
from datetime import datetime

import omero_tools as omero_tools
from microscopemetrics import samples
from omero.gateway import BlitzGateway, ImageWrapper

# Creating logging services
module_logger = logging.getLogger("microscopemetricsomero.analysis")

# Namespace constants
NAMESPACE_PREFIX = "microscope_metrics"
NAMESPACE_ANALYZED = "analyzed"
NAMESPACE_VALIDATED = "validated"
# TODO: Add a special case editable

ANALYSIS_CLASS_MAPPINGS = {
    "ArgolightBAnalysis": samples.argolight.ArgolightBAnalysis,
    "ArgolightEAnalysis": samples.argolight.ArgolightEAnalysis,
    "PSFBeadsAnalysis": samples.psf_beads.PSFBeadsAnalysis,
}


def generate_namespace(sections: list = [NAMESPACE_PREFIX, NAMESPACE_ANALYZED]) -> str:
    # TODO: get version directly from package
    return "/".join(sections)


def generate_analysis_annotation(start_time, end_time, analysis_config) -> dict:
    return {
        "analysis_class": analysis_config["analysis_class"],
        "start_time": str(start_time),
        "end_time": str(end_time),
        **analysis_config["parameters"],
    }


def _analyze_image(
    conn: BlitzGateway, image: ImageWrapper, analysis_config: dict
) -> None:
    start_time = datetime.now()
    # Create analysis instance
    analysis = ANALYSIS_CLASS_MAPPINGS[analysis_config["analysis_class"]]()
    analysis.set_data(
        analysis_config["data"]["name"], omero_tools.get_image_intensities(image)
    )
    for par, val in analysis_config["parameters"].items():
        analysis.set_metadata(par, val)

    module_logger.info(
        f"Running analysis {analysis_config['analysis_class']} on image {image.getId()}"
    )

    analysis.run()

    _dump_image_analysis(conn, image, analysis, analysis_config)

    module_logger.info(
        f"Analysis {analysis_config['analysis_class']} on image {image.getId()} completed"
    )

    end_time = datetime.now()
    module_logger.info("Annotating analysis metadata")
    omero_tools.create_key_value(
        conn=conn,
        annotation=generate_analysis_annotation(start_time, end_time, analysis_config),
        omero_object=image,
        annotation_name="microscope-metrics analysis metadata",
        namespace=generate_namespace(),
    )


def _dump_image_analysis(
    conn: BlitzGateway,
    image: ImageWrapper,
    analysis: microscopemetrics.samples.Analysis,
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
        object=object,
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


def analyze_dataset(conn, script_params, dataset, config):
    # TODO: must note in mapann the analyses that were done
    # TODO: do something to automate selection of microscope type

    module_logger.info(f"Analyzing data from Dataset: {dataset.getId()}")
    module_logger.info(f"Date and time: {datetime.now()}")

    for analysis_name, analysis_config in config["analyses_config"]["ANALYSES"].items():
        if analysis_config["do_analysis"]:
            module_logger.info(
                f"Running analysis {analysis_name} on sample {analysis_config['sample']}"
            )

            images = omero_tools.get_tagged_images_in_dataset(
                dataset, analysis_config["data"]["tag_id"]
            )

            for image in images:
                _analyze_image(conn=conn, image=image, config=analysis_config)
