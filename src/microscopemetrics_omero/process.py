import logging
from datetime import datetime
from typing import Union
from dataclasses import fields

from microscopemetrics_omero import omero_tools
from microscopemetrics.samples import argolight, field_illumination
from microscopemetrics.data_schema import core_schema as mm_schema
from omero.gateway import BlitzGateway, ImageWrapper, DatasetWrapper, ProjectWrapper

from microscopemetrics_omero import dump, load

logger = logging.getLogger(__name__)

ANALYSIS_CLASS_MAPPINGS = {
    "ArgolightBAnalysis": argolight.ArgolightBAnalysis,
    "ArgolightEAnalysis": argolight.ArgolightEAnalysis,
    "PSFBeadsAnalysis": field_illumination.FieldIlluminationAnalysis,
}

OBJECT_TO_DUMP_FUNCTION = {
    mm_schema.Image: dump.dump_image,
    mm_schema.ROI: dump.dump_roi,
    mm_schema.Tag: dump.dump_tag,
    mm_schema.KeyValues: dump.dump_key_value,
    mm_schema.Table: dump.dump_table,
}


def _annotate_processing(
    omero_object: Union[ImageWrapper, DatasetWrapper, ProjectWrapper],
    start_time: datetime.time,
    end_time: datetime.time,
    analysis_config: dict,
    namespace: str,
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
    logger.info(
        f"Running analysis {analysis_config['analysis_class']} on image: {image.getId()}"
    )
    # TODO: remove this
    logger.info(analysis_config)
    logger.info(f"at process_image: {type(image)}")

    analysis = ANALYSIS_CLASS_MAPPINGS[analysis_config["analysis_class"]](
        name=analysis_config["name"],
        description=analysis_config["description"],
        input={
            analysis_config["data"]["name"]: {
                "data": load.load_image(image),
                "name": image.getName(),
                "image_url": omero_tools.get_url_from_object(image),
            },
            **analysis_config["parameters"],
        },
        output={},
    )
    analysis.run()

    logger.info(
        f"Analysis {analysis_config['analysis_class']} on image {image.getId()} completed"
    )

    return analysis


def process_dataset(dataset: DatasetWrapper, config: dict) -> None:
    # TODO: must note in map_ann the analyses that were done
    # TODO: get comment from script_params
    # TODO: how to process multi-image analysis?

    logger.info(f"Analyzing data from Dataset: {dataset.getId()}")
    logger.info(config)

    for analysis_name, analysis_config in config["assay_config"]["analysis"].items():
        if analysis_config["do_analysis"]:
            logger.info(f"Running analysis {analysis_name}...")
            # TODO: verify if the analysis was already done
            start_time = datetime.now()

            images = omero_tools.get_tagged_images_in_dataset(
                dataset, analysis_config["data"]["tag_id"]
            )

            for image in images:  # TODO: This seems to cover only single image analysis
                logger.info(f"at process_dataset: {type(image)}")  # TODO: remove this
                mm_dataset = process_image(image=image, analysis_config=analysis_config)
                if not mm_dataset.processed:
                    logger.error("Analysis failed. Not dumping data")

                dump_image_process(
                    image=image,
                    analysis=mm_dataset,
                )

            try:
                logger.info("Adding comment")
                comment = config["script_parameters"]["Comment"]
            except KeyError:
                logger.info("No comment provided")
                comment = None
            if comment is not None:
                omero_tools.create_comment(
                    conn=dataset._conn,
                    omero_object=dataset,
                    comment_text=comment,
                    namespace=ANALYSIS_CLASS_MAPPINGS[
                        analysis_config["analysis_class"]
                    ].class_model_uri,
                )
            end_time = datetime.now()

            logger.info("Annotating processing metadata")
            _annotate_processing(
                omero_object=dataset,
                start_time=start_time,
                end_time=end_time,
                analysis_config=analysis_config,
                namespace=ANALYSIS_CLASS_MAPPINGS[
                    analysis_config["analysis_class"]
                ].class_model_uri,
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
    logger.info(f"Dumping {output_element.name}")
    conn = target_omero_object._conn
    if isinstance(output_element, list):
        for e in output_element:
            dump_output_element(e, target_omero_object)
    else:
        logger.info(f"Dumping {output_element.name} to OMERO")
        conn = target_omero_object._conn
        for t, f in OBJECT_TO_DUMP_FUNCTION.items():
            if isinstance(output_element, t):
                f(conn, output_element, target_omero_object)
                break
