import logging
from datetime import datetime



import omero_tools as omero_tools
from microscopemetrics import samples
from omero.gateway import BlitzGateway, ImageWrapper, DatasetWrapper, ProjectWrapper

from dump import dump_image_process

# Creating logging services
module_logger = logging.getLogger("microscopemetricsomero.analysis")

# Namespace constants
NAMESPACE_PREFIX = "microscopemetrics"
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

    dump_image_process(conn, image, analysis, analysis_config)

    module_logger.info(
        f"Analysis {analysis_config['analysis_class']} on image {image.getId()} completed"
    )

    end_time = datetime.now()
    module_logger.info("Annotating analysis metadata")
    omero_tools.create_key_value(
        conn=conn,
        annotation=generate_analysis_annotation(start_time, end_time, analysis_config),
        omero_object=image,
        start_time=start_time,
        end_time=end_time,
        analysis_config=analysis_config,
    )


def process_dataset(conn, script_params, dataset, config):
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
