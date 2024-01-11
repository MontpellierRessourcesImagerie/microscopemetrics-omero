import ezomero
import pytest
import yaml
from microscopemetrics_schema.datamodel import (
    ArgolightBDataset,
    ArgolightEDataset,
    FieldIlluminationDataset,
)
from omero.gateway import FileAnnotationWrapper

from microscopemetrics_omero import omero_tools, process


def _generate_dataset(
    conn, analysis_name, project_structure, dataset_name, image_name, ns, study_conf_path
):
    current_conn = conn.suConn("facility_manager_microscope_1", "microscope_1")

    dataset_info = project_structure["dataset_info"]
    dataset_id = dataset_info[dataset_name]
    dataset = current_conn.getObject("Dataset", dataset_id)
    image_info = project_structure["image_info"]
    image_id = image_info[image_name]
    image = current_conn.getObject("Image", image_id)
    project = dataset.getParent()
    project_id = project.getId()

    ezomero.post_file_annotation(
        current_conn,
        "Project",
        project_id,
        study_conf_path,
        ns=ns,
        across_groups=False,
    )
    tag = omero_tools.create_tag(
        current_conn, "target_image", tag_description="", omero_object=image
    )

    script_params = {"Comment": "This is a test comment"}
    with open("./src/microscopemetrics_omero/config/main_config.yaml", "r") as f:
        main_config = yaml.load(f, Loader=yaml.SafeLoader)
    for ann in project.listAnnotations():
        if (
            isinstance(ann, FileAnnotationWrapper)
            and ann.getFileName() == main_config["study_conf_file_name"]
        ):
            study_config = yaml.load(
                ann.getFileInChunks().__next__().decode(), Loader=yaml.SafeLoader
            )
            break
    study_config["analysis"][analysis_name]["data"]["tag_id"] = tag.getId()
    config = {
        "script_parameters": script_params,
        "main_config": main_config,
        "study_config": study_config,
    }

    return dataset, config


@pytest.fixture(scope="module")
def field_illumination_dataset(conn, project_structure):
    return _generate_dataset(
        conn,
        "FieldIllumination",
        project_structure,
        "fh_date_stamp_1",
        "field_illumination_image.czi",
        FieldIlluminationDataset.class_model_uri,
        "./tests/data/config_files/field_illumination/study_config.yaml",
    )


@pytest.fixture(scope="module")
def argolight_b_dataset(conn, project_structure):
    return _generate_dataset(
        conn,
        "ArgolightB",
        project_structure,
        "ab_date_stamp_1",
        "argolight_b_image.dv",
        ArgolightBDataset.class_model_uri,
        "./tests/data/config_files/argolight_b/study_config.yaml",
    )


@pytest.fixture(scope="module")
def argolight_e_xres_dataset(conn, project_structure):
    return _generate_dataset(
        conn,
        "Argolight_E_x_resolution",
        project_structure,
        "ae_date_stamp_1",
        "argolight_e_x-res_image.dv",
        ArgolightEDataset.class_model_uri,
        "./tests/data/config_files/argolight_e/study_config.yaml",
    )


@pytest.fixture(scope="module")
def argolight_e_yres_dataset(conn, project_structure):
    return _generate_dataset(
        conn,
        "Argolight_E_y_resolution",
        project_structure,
        "ae_date_stamp_1",
        "argolight_e_y-res_image.dv",
        ArgolightEDataset.class_model_uri,
        "./tests/data/config_files/argolight_e/study_config.yaml",
    )


def test_field_illumination(field_illumination_dataset):
    dataset, config = field_illumination_dataset
    process.process_dataset(dataset=dataset, config=config)

    process_annotation_ids = ezomero.get_map_annotation_ids(
        conn=dataset._conn,
        object_type="dataset",
        object_id=dataset.getId(),
        ns=str(FieldIlluminationDataset.class_model_uri),
        across_groups=False,
    )
    process_annotation = ezomero.get_map_annotation(
        conn=dataset._conn, map_ann_id=process_annotation_ids[-1], across_groups=False
    )

    assert process_annotation
    assert process_annotation["analysis_class"] == "FieldIlluminationAnalysis"


def test_argolight_b(argolight_b_dataset):
    dataset, config = argolight_b_dataset
    process.process_dataset(dataset=dataset, config=config)

    process_annotation_ids = ezomero.get_map_annotation_ids(
        conn=dataset._conn,
        object_type="dataset",
        object_id=dataset.getId(),
        ns=str(ArgolightBDataset.class_model_uri),
        across_groups=False,
    )
    process_annotation = ezomero.get_map_annotation(
        conn=dataset._conn, map_ann_id=process_annotation_ids[-1], across_groups=False
    )

    assert process_annotation
    assert process_annotation["analysis_class"] == "ArgolightBAnalysis"


def test_argolight_e_xres(argolight_e_xres_dataset):
    dataset, config = argolight_e_xres_dataset
    process.process_dataset(dataset=dataset, config=config)

    process_annotation_ids = ezomero.get_map_annotation_ids(
        conn=dataset._conn,
        object_type="dataset",
        object_id=dataset.getId(),
        ns=str(ArgolightEDataset.class_model_uri),
        across_groups=False,
    )
    process_annotation = ezomero.get_map_annotation(
        conn=dataset._conn, map_ann_id=process_annotation_ids[-1], across_groups=False
    )

    assert process_annotation
    assert process_annotation["analysis_class"] == "ArgolightEAnalysis"


def test_argolight_e_yres(argolight_e_yres_dataset):
    dataset, config = argolight_e_yres_dataset
    process.process_dataset(dataset=dataset, config=config)

    process_annotation_ids = ezomero.get_map_annotation_ids(
        conn=dataset._conn,
        object_type="dataset",
        object_id=dataset.getId(),
        ns=str(ArgolightEDataset.class_model_uri),
        across_groups=False,
    )
    process_annotation = ezomero.get_map_annotation(
        conn=dataset._conn, map_ann_id=process_annotation_ids[-1], across_groups=False
    )

    assert process_annotation
    assert process_annotation["analysis_class"] == "ArgolightEAnalysis"
