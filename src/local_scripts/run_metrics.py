import logging
from io import StringIO
from omero.gateway import BlitzGateway
import yaml

logger = logging.getLogger(__name__)
log_string = StringIO()
string_hdl = logging.StreamHandler(log_string)
string_hdl.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s: %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
string_hdl.setFormatter(formatter)
logger.addHandler(string_hdl)



def _read_config_from_file_ann(file_annotation):
    return yaml.load(
        file_annotation.getFileInChunks().__next__().decode(), Loader=yaml.SafeLoader
    )


def run_script_local():

    try:
        with open("../microscopemetrics_omero/config/main_config.yaml", "r") as f:
            main_config = yaml.load(f, Loader=yaml.SafeLoader)
    except FileNotFoundError:
        logger.error("No main configuration file found: Contact your administrator.")
        return


    conn = BlitzGateway(
        username="root", passwd="omero", group="system", port=6064, host="localhost", secure=True
    )

    script_params = {
        "IDs": [1],
        "Configuration file name": "yearly_config.yaml",
        "Comment": "This is a test comment",
    }

    try:
        conn.connect()

        logger.info(f"Metrics started using parameters: \n{script_params}")
        logger.info(f"Start time: {datetime.now()}")

        logger.info(f"Connection successful: {conn.isConnected()}")

        # Getting the configuration files
        assay_config =
        analysis_config = MetricsConfig()
        analysis_config.read("main_config.ini")  # TODO: read main config from somewhere
        analysis_config.read(script_params["Configuration file name"])

        device_config = MetricsConfig()
        device_config.read(analysis_config.get("MAIN", "device_conf_file_name"))

        datasets = conn.getObjects(
            "Dataset", script_params["IDs"]
        )

        for dataset in datasets:
            microscope_prj = dataset.getParent()  # We assume one project per dataset

            if microscope_prj is None:
                logger.error(
                    f"No parent project found for dataset {dataset.getName()}: "
                    f"Every dataset must be part of a project and only one project."
                )
                continue

            assay_conf_file_name = f"{script_params['Assay type']}_config.yaml"
            assay_config = None
            for ann in microscope_prj.listAnnotations():
                if (
                    type(ann) == gateway.FileAnnotationWrapper
                    and ann.getFileName() == assay_conf_file_name
                ):
                    assay_config = _read_config_from_file_ann(ann)
            if not assay_config:
                logger.error(
                    f"No assay configuration {assay_conf_file_name} found for dataset {dataset.getName()}: "
                    f"Please contact your administrator"
                )
                continue

            config = {
                "script_parameters": script_params,
                "main_config": main_config,
                "assay_config": assay_config,
            }
            process.process_dataset(dataset=dataset, config=config)

        logger.info(f"Metrics analysis finished")

    finally:
        logger.info("Closing connection")
        print(log_string.getvalue())
        log_string.close()
        conn.close()

