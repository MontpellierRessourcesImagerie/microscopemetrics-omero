from microscopemetrics_omero import process
import pytest
import ezomero

@pytest.fixture
def field_illumination_dataset(conn, project_structure):

    current_conn = conn.suConn("facility_manager_microscope_1", "microscope_1")

    project_info = project_structure["project_info"]
    pr_id = project_info["Field_homogeneity"]
    project = conn.getObject("Project", pr_id)
    dataset_info = project_structure["dataset_info"]
    ds_id = dataset_info["date_stamp_1"]
    dataset = conn.getObject("Dataset", ds_id)

    ezomero.post_file_annotation(current_conn, "Project", pr_id,
                                 "./tests/data/config_files/field_illumination_config.yaml",
                                 )

    # TODO: tag image with field illumination tag

    return dataset


def test_field_illumination(field_illumination_dataset):
    process.field_illumination(field_illumination_dataset)
    assert False