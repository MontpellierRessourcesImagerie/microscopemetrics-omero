import logging
from datetime import datetime
from typing import Union
from dataclasses import fields

from microscopemetrics_omero import omero_tools
from microscopemetrics.samples import (
    argolight,
    field_illumination
)
from microscopemetrics.data_schema import core_schema as mm_schema
from omero.gateway import BlitzGateway, ImageWrapper, DatasetWrapper, ProjectWrapper

from microscopemetrics_omero import dump, load

# Creating logging services
module_logger = logging.getLogger("microscopemetrics_omero.process")


ANALYSIS_CLASS_MAPPINGS = {
    "ArgolightBAnalysis": argolight.ArgolightBAnalysis,
    "ArgolightEAnalysis": argolight.ArgolightEAnalysis,
    "PSFBeadsAnalysis": field_illumination.FieldIlluminationAnalysis,
}

INPUT_TYPE_TO_LOAD_FUNCTION = {
    mm_schema.Image: load.load_image,
    mm_schema.Dataset: load.load_dataset,
    mm_schema.Project: load.load_project,
}


def _annotate_processing(omero_object: Union[ImageWrapper, DatasetWrapper, ProjectWrapper],
                         start_time: datetime.time,
                         end_time: datetime.time,
                         analysis_config: dict,
                         namespace: str
                         ) -> None:
    annotation = {
        "analysis_class": analysis_config["analysis_class"],
        "start_time": str(start_time),
        "end_time": str(end_time),
        **analysis_config["parameters"],
    }

    omero_tools.create_key_value(
        conn=omero_object._conn,
        annotation=annotation,
        omero_object=omero_object,
        annotation_name="microscopemetrics processing metadata",
        namespace=namespace,
    )


def process_image(
    image: ImageWrapper, analysis_config: dict
) -> mm_schema.MetricsDataset:
    analysis = ANALYSIS_CLASS_MAPPINGS[analysis_config["analysis_class"]](
        name=analysis_config["name"],
        description=analysis_config["description"],
        input={
            analysis_config["data"]["name"]: {
                "data": omero_tools.get_image_intensities(image),
                "name": image.getName(),
                "image_url": omero_tools.get_url_from_object(image),
            },
            **analysis_config["parameters"],
        },
        output={},
    )
    module_logger.info(f"Running analysis {analysis.class_name} on image: {image.getId()}")

    analysis.run()

    module_logger.info(f"Analysis {analysis_config['analysis_class']} on image {image.getId()} completed")

    return analysis


def process_dataset(
        script_params: dict,
        dataset: DatasetWrapper,
        config: dict
) -> None:
    # TODO: must note in map_ann the analyses that were done
    # TODO: get comment from script_params
    # TODO: how to process multi-image analysis?

    module_logger.info(f"Analyzing data from Dataset: {dataset.getId()}")

    for analysis_name, analysis_config in config["analyses_config"]["assays"].items():
        if analysis_config["do_analysis"]:
            module_logger.info(
                f"Running analysis {analysis_name}..."
            )
            # TODO: verify if the analysis was already done

            images = omero_tools.get_tagged_images_in_dataset(
                dataset, analysis_config["data"]["tag_id"]
            )

            for image in images:
                start_time = datetime.now()

                mm_dataset = process_image(image=image, analysis_config=analysis_config)
                if not mm_dataset.processed:
                    module_logger.error("Analysis failed. Not dumping data")

                dump_image_process(
                    image=image,
                    analysis=mm_dataset,
                )
                end_time = datetime.now()

                module_logger.info("Annotating processing metadata")
                _annotate_processing(
                    omero_object=image,
                    start_time=start_time,
                    end_time=end_time,
                    analysis_config=analysis_config,
                    namespace=mm_dataset.class_model_uri,
                )


def dump_image_process(
    image: ImageWrapper,
    analysis: mm_schema.MetricsDataset,
) -> None:
    for output_field in fields(analysis.output):
        output_element = getattr(analysis.output, output_field.name)
        dump_output_element(
            output_element=output_element,
            target_omero_object=image,
        )


def dump_output_element(
    output_element: mm_schema.MetricsOutput,
    target_omero_object: Union[ImageWrapper, DatasetWrapper, ProjectWrapper],
) -> None:
    module_logger.info(f"Dumping {output_element.name}")
    conn = target_omero_object._conn
    if isinstance(output_element, list):
        [dump_output_element(conn, e, target_omero_object) for e in output_element]
    elif isinstance(output_element, mm_schema.Image):
        dump.dump_image(conn, output_element, target_omero_object)
    elif isinstance(output_element, mm_schema.ROI):
        dump.dump_roi(conn, output_element, target_omero_object)
    elif isinstance(output_element, mm_schema.Tag):
        dump.dump_tag(conn, output_element, target_omero_object)
    elif isinstance(output_element, mm_schema.KeyValues):
        dump.dump_key_value(conn, output_element, target_omero_object)
    elif isinstance(output_element, mm_schema.Table):
        dump.dump_table(conn, output_element, target_omero_object)

    # TODO: make this a match statement
    # match output_element:
    #     case list():
    #         [dump_output_element(conn, e, target_omero_object, namespace) for e in output_element]
    #     case mm_schema.Image():
    #         dump.dump_image(conn, output_element, target_omero_object)
    #     case mm_schema.ROI():
    #         dump.dump_roi(conn, output_element, target_omero_object)
    #     case mm_schema.Tag():
    #         dump.dump_tag(conn, output_element, target_omero_object)
    #     case mm_schema.KeyValues():
    #         dump.dump_key_value(conn, output_element, target_omero_object)
    #     case mm_schema.Table():
    #         dump.dump_table(conn, output_element, target_omero_object)
    #     case mm_schema.Comment():
    #         dump.dump_comment(conn, output_element, target_omero_object)
    #

