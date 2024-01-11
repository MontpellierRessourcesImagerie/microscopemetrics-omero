import logging
from datetime import datetime
from io import StringIO

import yaml
from omero.gateway import BlitzGateway, FileAnnotationWrapper

from microscopemetrics_omero import process

logger = logging.getLogger(__name__)
log_string = StringIO()
string_hdl = logging.StreamHandler(log_string)
string_hdl.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s: %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
string_hdl.setFormatter(formatter)
logger.addHandler(string_hdl)


def _get_config_from_file_ann(omero_object, file_name):
    for ann in omero_object.listAnnotations():
        if type(ann) == FileAnnotationWrapper and ann.getFileName() == file_name:
            return yaml.load(ann.getFileInChunks().__next__().decode(), Loader=yaml.SafeLoader)

    logger.error(
        f"No assay configuration {file_name} found for dataset {omero_object.getName()}: "
        f"Please contact your administrator"
    )
    return None


def run_script_local():
    try:
        with open("../microscopemetrics_omero/config/main_config.yaml", "r") as f:
            main_config = yaml.load(f, Loader=yaml.SafeLoader)
    except FileNotFoundError:
        logger.error("No main configuration file found: Contact your administrator.")
        return

    conn = BlitzGateway(
        username="facility_manager_microscope_1",
        passwd="abc123",
        group="microscope_1",
        port=6064,
        host="localhost",
        secure=True,
    )

    # script_params = {
    #     "IDs": [1],
    #     "Comment": "This is a test comment",
    # }
    script_params = {
        "IDs": [2],
        "Comment": "This is a test comment",
    }

    try:
        conn.connect()

        logger.info(f"Metrics started using parameters: \n{script_params}")
        logger.info(f"Start time: {datetime.now()}")

        logger.info(f"Connection successful: {conn.isConnected()}")

        datasets = conn.getObjects("Dataset", script_params["IDs"])

        for dataset in datasets:
            microscope_prj = dataset.getParent()  # We assume one project per dataset

            if microscope_prj is None:
                logger.error(
                    f"No parent project found for dataset {dataset.getName()}: "
                    f"Every dataset must be part of a project and only one project."
                )
                continue

            study_conf_file_name = main_config["study_conf_file_name"]
            study_config = _get_config_from_file_ann(microscope_prj, study_conf_file_name)

            config = {
                "script_parameters": script_params,
                "main_config": main_config,
                "study_config": study_config,
            }
            process.process_dataset(dataset=dataset, config=config)

        logger.info(f"Metrics analysis finished")

    finally:
        logger.info("Closing connection")
        print(log_string.getvalue())
        log_string.close()
        conn.close()


if __name__ == "__main__":
    run_script_local()
