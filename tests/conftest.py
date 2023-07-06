import os
import pytest
import subprocess
import numpy as np
from datetime import datetime
from omero.cli import CLI
from omero.gateway import BlitzGateway
from omero.gateway import ScreenWrapper, PlateWrapper
from omero.model import ScreenI, PlateI, WellI, WellSampleI, ImageI
from omero.model import ScreenPlateLinkI
from omero.plugins.sessions import SessionsControl
from omero.plugins.user import UserControl
from omero.plugins.group import GroupControl
from omero.rtypes import rint
import importlib.util
import pandas as pd

from microscopemetrics.samples import Analysis, model
from pydantic.color import Color


import ezomero


# Settings for OMERO
DEFAULT_OMERO_USER = "root"
DEFAULT_OMERO_PASS = "omero"
DEFAULT_OMERO_HOST = "localhost"
DEFAULT_OMERO_WEB_HOST = "http://localhost:5080"
DEFAULT_OMERO_PORT = 6064
DEFAULT_OMERO_SECURE = 1

# [[group, permissions], ...]
GROUPS_TO_CREATE = [
    ["microscope_1_group", "read-only"],
    ["microscope_2_group", "read-only"],
    ["regular_user_group", "read-only"],
]

# [[user, [groups to be added to], [groups to own]], ...]
USERS_TO_CREATE = [
    [
        "facility_manager_microscope_1",
        ["microscope_1_group", "microscope_2_group"],
        ["microscope_1_group"],
    ],
    [
        "facility_manager_microscope_2",
        ["microscope_1_group", "microscope_2_group"],
        ["microscope_2_group"],
    ],
    ["regular_user", ["regular_user_group"], []],
]


def pytest_addoption(parser):
    parser.addoption(
        "--omero-user",
        action="store",
        default=os.environ.get("OMERO_USER", DEFAULT_OMERO_USER),
    )
    parser.addoption(
        "--omero-pass",
        action="store",
        default=os.environ.get("OMERO_PASS", DEFAULT_OMERO_PASS),
    )
    parser.addoption(
        "--omero-host",
        action="store",
        default=os.environ.get("OMERO_HOST", DEFAULT_OMERO_HOST),
    )
    parser.addoption(
        "--omero-web-host",
        action="store",
        default=os.environ.get("OMERO_WEB_HOST", DEFAULT_OMERO_WEB_HOST),
    )
    parser.addoption(
        "--omero-port",
        action="store",
        type=int,
        default=int(os.environ.get("OMERO_PORT", DEFAULT_OMERO_PORT)),
    )
    parser.addoption(
        "--omero-secure",
        action="store",
        default=bool(os.environ.get("OMERO_SECURE", DEFAULT_OMERO_SECURE)),
    )


class TestAnalysis(Analysis):
    """A class implementing all analysis options for testing."""

    def __init__(self):
        super().__init__(output_description="This analysis is for testing...")

        self.add_data_requirement(
            name="image_test_requirement",
            description="A image for testing",
            data_type=np.ndarray,
        )
        self.add_metadata_requirement(
            name="metadata_float_test_requirement",
            description="A required float value",
            data_type=float,
            units="MICRON",
            optional=False,
        )
        self.add_metadata_requirement(
            name="metadata_float_test_requirement_optional",
            description="An optional float value",
            data_type=float,
            units="MICRON",
            optional=True,
        )
        self.add_metadata_requirement(
            name="metadata_int_test_requirement",
            description="A required int value",
            data_type=int,
            optional=False,
        )
        self.add_metadata_requirement(
            name="metadata_str_test_requirement",
            description="A required str value",
            data_type=str,
            optional=False,
        )

    def _run(self):
        test_image = np.zeros((200, 201, 20, 3, 1), dtype=np.uint8)
        test_image[100:0, 100:0, 0:10, 0, :] = 255
        test_image[100:0, 100:0, 11:20, 1, :] = 255
        test_image[101:200, 101:201, :, 2, :] = 255

        self.output.append(model.Image(data=test_image, name="test_output_image", description="An output image for testing"))

        shapes = [
            model.Point(x=10, y=10, stroke_color=Color("red")),
            model.Line(x1=5, y1=5, x2=15, y2=20, stroke_color=Color("green")),
            model.Rectangle(x=5, y=5, w=30, h=35, stroke_color=Color("blue")),
            model.Ellipse(x=5, y=5, x_rad=30, y_rad=35, stroke_color=Color("yellow")),
            model.Polygon(points=[(5, 5), (15, 5), (15, 15), (5, 15)], stroke_color=Color("orange")),
        ]
        self.output.append(
            model.Roi(
                name="test_output_roi",
                description="An output ROI for testing",
                shapes=shapes,
            )
        )

        self.output.append(
            model.Tag(
                name="test_output_tag",
                description="An output tag for testing",
                tag_value="test",
            )
        )

        self.output.append(
            model.KeyValues(
                name="test_output_key_values",
                description="An output key-value store for testing",
                key_values={"a": 1, "b": 2, "c": 3},
            )
        )

        self.output.append(
            model.Table(
                name="test_output_table",
                description="An output table for testing",
                table=pd.DataFrame(
                    {
                        "a": [1, 2, 3],
                        "b": [4, 5, 6],
                        "c": [7, 8, 9],
                    }
                )
            )
        )

        self.output.append(
            model.Comment(
                name="test_output_comment",
                description="An output comment for testing",
                comment="test",
            )
        )

        return True


# we can change this later
@pytest.fixture(scope="session")
def omero_params(request):
    user = request.config.getoption("--omero-user")
    password = request.config.getoption("--omero-pass")
    host = request.config.getoption("--omero-host")
    web_host = request.config.getoption("--omero-web-host")
    port = request.config.getoption("--omero-port")
    secure = request.config.getoption("--omero-secure")
    return user, password, host, web_host, port, secure


@pytest.fixture(scope="session")
def users_groups(conn, omero_params):
    session_uuid = conn.getSession().getUuid().val
    user = omero_params[0]
    host = omero_params[2]
    port = str(omero_params[4])
    cli = CLI()
    cli.register("sessions", SessionsControl, "test")
    cli.register("user", UserControl, "test")
    cli.register("group", GroupControl, "test")

    group_info = []
    for gname, gperms in GROUPS_TO_CREATE:
        cli.invoke(
            [
                "group",
                "add",
                gname,
                "--type",
                gperms,
                "-k",
                session_uuid,
                "-u",
                user,
                "-s",
                host,
                "-p",
                port,
            ]
        )
        gid = ezomero.get_group_id(conn, gname)
        group_info.append([gname, gid])

    user_info = []
    for user, groups_add, groups_own in USERS_TO_CREATE:
        # make user while adding to first group
        cli.invoke(
            [
                "user",
                "add",
                user,
                "test",
                "tester",
                "--group-name",
                groups_add[0],
                "-e",
                "useremail@cnrs.fr",
                "-P",
                "abc123",
                "-k",
                session_uuid,
                "-u",
                user,
                "-s",
                host,
                "-p",
                port,
            ]
        )

        # add user to rest of groups
        if len(groups_add) > 1:
            for group in groups_add[1:]:
                cli.invoke(
                    [
                        "group",
                        "adduser",
                        "--user-name",
                        user,
                        "--name",
                        group,
                        "-k",
                        session_uuid,
                        "-u",
                        user,
                        "-s",
                        host,
                        "-p",
                        port,
                    ]
                )

        # make user owner of listed groups
        if len(groups_own) > 0:
            for group in groups_own:
                cli.invoke(
                    [
                        "group",
                        "adduser",
                        "--user-name",
                        user,
                        "--name",
                        group,
                        "--as-owner",
                        "-k",
                        session_uuid,
                        "-u",
                        user,
                        "-s",
                        host,
                        "-p",
                        port,
                    ]
                )
        uid = ezomero.get_user_id(conn, user)
        user_info.append([user, uid])

    return group_info, user_info


@pytest.fixture(scope="session")
def project_structure(conn, timestamp, image_fixture, users_groups, omero_params):
    group_info, user_info = users_groups
    # Don't change anything for default_user!
    # If you change anything about users/groups, make sure they exist
    # [[group, [projects]], ...] per user
    project_str = {
        "users": [
            {
                "name": "default_user",
                "groups": [
                    {
                        "name": "default_group",
                        "projects": [
                            {
                                "name": f"proj0_{timestamp}",
                                "datasets": [
                                    {
                                        "name": f"ds0_{timestamp}",
                                        "images": [f"im0_{timestamp}"],
                                    }
                                ],
                            }
                        ],
                        "datasets": [],
                    }
                ],
            },
            {
                "name": "facility_manager_microscope_1",
                "groups": [
                    {
                        "name": "microscope_1_group",
                        "projects": [
                            {
                                "name": f"proj1_{timestamp}",
                                "datasets": [
                                    {
                                        "name": f"ds1_{timestamp}",
                                        "images": [f"im1_{timestamp}"],
                                    }
                                ],
                            },
                            {"name": f"proj2_{timestamp}", "datasets": []},
                        ],
                        "datasets": [],
                    },
                ],
            },
            {
                "name": "facility_manager_microscope_2",
                "groups": [
                    {
                        "name": "microscope_2_group",
                        "projects": [
                            {
                                "name": f"proj4_{timestamp}",
                                "datasets": [
                                    {
                                        "name": f"ds4_{timestamp}",
                                        "images": [f"im5_{timestamp}"],
                                    }
                                ],
                            },
                            {
                                "name": f"proj5_{timestamp}",
                                "datasets": [
                                    {"name": f"ds5_{timestamp}", "images": []}
                                ],
                            },
                        ],
                        "datasets": [],
                    },
                ],
            },
        ]
    }
    project_info = []
    dataset_info = []
    image_info = []
    for user in project_str["users"]:
        username = user["name"]
        for group in user["groups"]:
            groupname = group["name"]
            current_conn = conn

            # New connection if user and group need to be specified
            if username != "default_user":
                current_conn = conn.suConn(username, groupname)

            # Loop to post projects, datasets, and images
            for project in group["projects"]:
                projname = project["name"]
                proj_id = ezomero.post_project(current_conn, projname, "test project")
                project_info.append([projname, proj_id])

                for dataset in project["datasets"]:
                    dsname = dataset["name"]
                    ds_id = ezomero.post_dataset(
                        current_conn, dsname, proj_id, "test dataset"
                    )
                    dataset_info.append([dsname, ds_id])

                    for imname in dataset["images"]:
                        im_id = ezomero.post_image(
                            current_conn, image_fixture, imname, dataset_id=ds_id, dim_order="zctyx"
                        )
                        image_info.append([imname, im_id])
            for dataset in group["datasets"]:
                dsname = dataset["name"]
                ds_id = ezomero.post_dataset(
                    current_conn, dsname, description="test dataset"
                )
                dataset_info.append([dsname, ds_id])

            # Close temporary connection if it was created
            if username != "default_user":
                current_conn.close()

    yield [project_info, dataset_info, image_info]
    current_group = conn.getGroupFromContext().getId()
    conn.SERVICE_OPTS.setOmeroGroup(-1)
    for pname, pid in project_info:
        conn.deleteObjects(
            "Project", [pid], deleteAnns=True, deleteChildren=True, wait=True
        )
    conn.SERVICE_OPTS.setOmeroGroup(current_group)


@pytest.fixture(scope="session")
def conn(omero_params):
    user, password, host, web_host, port, secure = omero_params
    conn = BlitzGateway(user, password, host=host, port=port, secure=secure)
    conn.connect()
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def image_fixture():  # format zctxy TODO: change to this format across teh board
    test_image = np.zeros((20, 3, 1, 200, 201), dtype=np.uint8)
    test_image[0:10, 0, :, 0:100, 0:100] = 255
    test_image[11:20, 1, :, 0:100, 0:100] = 255
    test_image[:, 2, :, 101:200, 101:201] = 255
    return test_image


@pytest.fixture(scope="session")
def timestamp():
    return f"{datetime.now():%Y%m%d%H%M%S}"


@pytest.fixture(scope="session")
def microscopemetrics_finished_analysis(image_fixture):
    mm_analysis = TestAnalysis()
    mm_analysis.set_data("image_test_requirement", image_fixture)
    mm_analysis.set_metadata("metadata_float_test_requirement", 1.0)
    mm_analysis.set_metadata("metadata_float_test_requirement_optional", 2.0)
    mm_analysis.set_metadata("metadata_int_test_requirement", 1)
    mm_analysis.set_metadata("metadata_str_test_requirement", "test")

    mm_analysis.run()

    return mm_analysis
