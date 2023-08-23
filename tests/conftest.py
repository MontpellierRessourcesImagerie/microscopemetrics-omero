import os
import pytest
import subprocess
import numpy as np
from datetime import datetime
from omero.cli import CLI
from omero.gateway import BlitzGateway
from omero.plugins.sessions import SessionsControl
from omero.plugins.user import UserControl
from omero.plugins.group import GroupControl
from omero.rtypes import rint
import importlib.util
import pandas as pd

import microscopemetrics.data_schema.core_schema as mm_schema
from microscopemetrics.samples import numpy_to_inlined_image, numpy_to_inlined_mask, dict_to_inlined_table

# TODO: make a test dataset programmatically
from microscopemetrics.samples.field_illumination import FieldIlluminationAnalysis
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
    ["microscope_1_group", "read-annotate"],
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


@pytest.fixture
def mm_finished_analysis():
    image_url = "https://dev.mri.cnrs.fr/attachments/download/2926/chroma.npy"
    data = np.ones((1, 1, 512, 512, 3))
    analysis = FieldIlluminationAnalysis(
        name="an analysis",
        description="a description",
        input={
            "field_illumination_image": {
                "data": data,
                "name": "image_name",
                "image_url": image_url,
            }
        },
        output={},
    )

    analysis.run()

    return analysis


@pytest.fixture
def mm_image_as_numpy_fixture(numpy_image_fixture):
    image_as_numpy = mm_schema.ImageAsNumpy(
        name="test_image",
        description="test image",
        image_url="https://example.com/image001",
        data=numpy_image_fixture,
    )

    return image_as_numpy


@pytest.fixture
def mm_image2d_fixture(numpy_image_fixture):
    image_2d = numpy_to_inlined_image(
        array=numpy_image_fixture[0, 0, :, :, 0],
        name="test_image",
        description="test image",
        image_url="https://example.com/image001",
        source_image_url="https://example.com/source_image001",
    )

    return image_2d


@pytest.fixture
def mm_image5d_fixture(numpy_image_fixture):
    image_5d = numpy_to_inlined_image(
        array=numpy_image_fixture,
        name="test_image",
        description="test image",
        image_url="https://example.com/image001",
        source_image_url="https://example.com/source_image001",
    )

    return image_5d


@pytest.fixture
def mm_image_mask_fixture():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True

    image_mask = numpy_to_inlined_mask(
        array=mask,
        name="test_mask",
        description="test mask",
        image_url="https://example.com/image001",
        source_image_url="https://example.com/source_image001",
    )

    return image_mask


@pytest.fixture
def mm_roi_fixture(mm_image_mask_fixture):
    shapes = [
        mm_schema.Point(
            label="point",
            x=10,
            y=10,
            z=0,
            t=0,
            c=0,
            fill_color=mm_schema.Color(r=255, g=0, b=0, alpha=128),
            stroke_color=mm_schema.Color(r=255, g=0, b=0, alpha=0),
        ),
        mm_schema.Line(
            label="line",
            x1=20,
            y1=20,
            x2=100,
            y2=100,
            z=0,
            t=0,
            c=0,
            fill_color=mm_schema.Color(r=0, g=255, b=0, alpha=128),
            stroke_color=mm_schema.Color(r=0, g=255, b=0, alpha=0),
            stroke_width=1,
        ),
        mm_schema.Rectangle(
            label="rectangle",
            x=50,
            y=50,
            w=100,
            h=100,
            z=0,
            t=0,
            c=0,
            fill_color=mm_schema.Color(r=0, g=0, b=255, alpha=128),
            stroke_color=mm_schema.Color(r=0, g=0, b=255, alpha=0),
            stroke_width=3,
        ),
        mm_schema.Ellipse(
            label="ellipse",
            x=100,
            y=100,
            x_rad=50,
            y_rad=50,
            z=0,
            t=0,
            c=0,
            fill_color=mm_schema.Color(r=128, g=128, b=0, alpha=128),
            stroke_color=mm_schema.Color(r=128, g=128, b=0, alpha=0),
            stroke_width=6,
        ),
        mm_schema.Polygon(
            label="polygon",
            vertexes=[
                mm_schema.Vertex(x=10, y=10),
                mm_schema.Vertex(x=20, y=20),
                mm_schema.Vertex(x=30, y=10),
                mm_schema.Vertex(x=20, y=5),
            ],
            is_open=False,
            z=0,
            t=0,
            c=0,
            fill_color=mm_schema.Color(r=0, g=128, b=128, alpha=128),
            stroke_color=mm_schema.Color(r=0, g=128, b=128, alpha=0),
            stroke_width=4,
        ),
        mm_schema.Mask(
            label="mask",
            mask=mm_image_mask_fixture,
            x=135,
            y=50,
            z=0,
            t=0,
            c=0,
        )
    ]

    roi = mm_schema.ROI(
        label="test_roi",
        image=["https://example.com/image001"],
        shapes=shapes,
        description="test roi",
    )

    return roi


@pytest.fixture
def mm_tag_fixture():
    tag = mm_schema.Tag(
        id=64,
        text="a test tag",
        description="test tag description",
    )

    return tag


@pytest.fixture
def mm_key_values_fixture():
    key_values = mm_schema.KeyValues(
        key_1=64,
        key_2="a test key",
        key_3=[1.0, 2.2, 3.3],
    )

    return key_values


@pytest.fixture
def mm_comment_fixture():
    comment = mm_schema.Comment(
        text="a test comment",
    )

    return comment


@pytest.fixture
def mm_table_as_pandas_df_fixture(pandas_df_fixture):
    table = mm_schema.TableAsPandasDF(
        df=pandas_df_fixture,
        name="test_table",
        description="test table description",
    )

    return table


@pytest.fixture
def mm_table_as_dict_fixture(dict_table_fixture):
    table = dict_to_inlined_table(
        dictionary=dict_table_fixture,
        name="test_table",
        description="test table description",
    )

    return table


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
def project_structure(conn, timestamp, numpy_image_fixture, users_groups, omero_params):
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
                            current_conn, numpy_image_fixture, imname, dataset_id=ds_id, dim_order="tzyxc"
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
def numpy_image_fixture():  # format tzyxc TODO: change to this format across the board
    test_image = np.zeros((1, 20, 200, 250, 3), dtype=np.uint8)
    test_image[:, 0:10, 0:100, 0:100, 0] = 255
    test_image[:, 11:20, 0:100, 0:100, 1] = 255
    test_image[:, :, 101:200, 125:250, 2] = 255

    return test_image


@pytest.fixture
def dict_table_fixture():
    table = {
        "str_data": ["string_01", "string_02"],
        "int_data": [1, 2],
        "float_data": [1.0, 2.0],
        # "array_float_data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # unsupported
        # "project_id": [1, 2],  # the projects column does not exist
        "dataset_id": [1, 2],
        "image_id": [1, 2],
        "roi_id": [1, 2],
    }

    return table


@pytest.fixture
def pandas_df_fixture(dict_table_fixture):
    df = pd.DataFrame.from_records(
        [dict(zip(dict_table_fixture, r)) for r in zip(*dict_table_fixture.values())]
    )

    return df


@pytest.fixture(scope="session")
def timestamp():
    return f"{datetime.now():%Y%m%d%H%M%S}"

