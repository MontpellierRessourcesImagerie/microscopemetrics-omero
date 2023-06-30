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
                            current_conn, image_fixture, imname, dataset_id=ds_id
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
def image_fixture():
    test_image = np.zeros((200, 201, 20, 3, 1), dtype=np.uint8)
    test_image[0:100, 0:100, 0:10, 0, :] = 255
    test_image[0:100, 0:100, 11:20, 1, :] = 255
    test_image[101:200, 101:201, :, 2, :] = 255
    return test_image


@pytest.fixture(scope="session")
def timestamp():
    return f"{datetime.now():%Y%m%d%H%M%S}"
