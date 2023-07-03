#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------
  Copyright (C) 2023 CNRS. All rights reserved.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along
  with this program; if not, write to the Free Software Foundation, Inc.,
  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

------------------------------------------------------------------------------

This script runs microscope-metrics on the selected dataset.

@author Julio Mateos Langerak
<a href="mailto:julio.mateos-langerak@igh.cnrs.fr">julio.mateos-langerak@igh.cnrs.fr</a>
@version Alpha0.1
<small>
(<b>Internal version:</b> $Revision: $Date: $)
</small>
@since 3.0-Beta4.3
"""

# import logging utilities
import logging
from datetime import datetime
from io import StringIO

import omero.gateway as gateway

# import omero dependencies
import omero.scripts as scripts

# import configuration parser
import yaml
from omero.rtypes import rlong, rstring

# import metrics
import src.microscopemetrics_omeroomero.analysis as analysis

# Creating logging services
logger = logging.getLogger("microscopemetrics")
logger.setLevel(logging.DEBUG)

# create a log string to return warnings in the web interface
log_string = StringIO()
string_hdl = logging.StreamHandler(log_string)
string_hdl.setLevel(logging.WARNING)
formatter = logging.Formatter(
    "%(asctime)s: %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
# formatter = logging.Formatter("%Y-%m-%d %H:%M:%S - %(levelname)s - %(message)s")
string_hdl.setFormatter(formatter)

# create console handler with a higher log level
console_hdl = logging.StreamHandler()
console_hdl.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_hdl.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(console_hdl)
logger.addHandler(string_hdl)


def run_script_local():
    from credentials import GROUP, HOST, PASSWORD, PORT, USER

    conn = gateway.BlitzGateway(
        username=USER, passwd=PASSWORD, group=GROUP, port=PORT, host=HOST
    )

    script_params = {
        # "IDs": [154],
        "IDs": [1],
        "Configuration file name": "yearly_config.ini",
        "Comment": "This is a test comment",
    }

    try:
        conn.connect()

        logger.info(f"Metrics started using parameters: \n{script_params}")
        logger.info(f"Start time: {datetime.now()}")

        logger.info(f"Connection successful: {conn.isConnected()}")

        # Getting the configuration files
        analysis_config = MetricsConfig()
        analysis_config.read("main_config.ini")  # TODO: read main config from somewhere
        analysis_config.read(script_params["Configuration file name"])

        device_config = MetricsConfig()
        device_config.read(analysis_config.get("MAIN", "device_conf_file_name"))

        datasets = conn.getObjects(
            "Dataset", script_params["IDs"]
        )  # generator of datasets

        for dataset in datasets:
            process.analyze_dataset(
                conn=conn,
                script_params=script_params,
                dataset=dataset,
                analyses_config=analysis_config,
                device_config=device_config,
            )

        print(log_string.getvalue())

    finally:
        logger.info("Closing connection")
        log_string.close()
        conn.close()


def _read_config_from_file_ann(file_annotation):
    return yaml.load(
        file_annotation.getFileInChunks().__next__().decode(), Loader=yaml.SafeLoader
    )


def run_script():
    client = scripts.client(
        "Run_Metrics.py",
        """This is the main script of microscope-metrics-omero. It will run the analyses on the selected 
        dataset. For more information check \n
        http://www.mri.cnrs.fr\n  
        Copyright: Write here some copyright info""",  # TODO: copyright info and documentation
        scripts.String(
            "Data_Type",
            optional=False,
            grouping="1",
            description="The dataset you want to work with.",
            values=[rstring("Dataset")],
            default="Dataset",
        ),
        scripts.List(
            "IDs", optional=False, grouping="1", description="List of Dataset IDs"
        ).ofType(rlong(0)),
        scripts.String(  # TODO: make enum list with other options. This should be in the main config file
            "Configuration file name",
            optional=False,
            grouping="1",
            default="monthly_config.ini",
            description="Add here any eventuality that you want to add to the analysis",
        ),
        scripts.String(
            "Comment",
            optional=True,
            grouping="2",
            description="Add here any eventuality that you want to add to the analysis",
        ),
    )

    try:
        script_params = {}
        for key in client.getInputKeys():
            if client.getInput(key):
                script_params[key] = client.getInput(key, unwrap=True)

        logger.info(f"Metrics started using parameters: \n{script_params}")
        logger.info(f"Start time: {datetime.now()}")

        with open("/etc/omero_metrics/main_config.yaml", "r") as f:
            main_config = yaml.load(f, Loader=yaml.SafeLoader)

        conn = gateway.BlitzGateway(client_obj=client)

        logger.info(f"Connection success: {conn.isConnected()}")

        datasets = conn.getObjects(
            "Dataset", script_params["IDs"]
        )  # generator of datasets

        for dataset in datasets:
            # Get the project / microscope
            microscope_prj = dataset.getParent()  # We assume one project per dataset

            device_config_file_name = main_config["MAIN"]["device_conf_file_name"]
            for ann in microscope_prj.listAnnotations():
                if type(ann) == gateway.FileAnnotationWrapper:
                    if ann.getFileName() == script_params["Configuration file name"]:
                        analyses_config = _read_config_from_file_ann(ann)
                    elif ann.getFileName() == device_config_file_name:
                        device_config = _read_config_from_file_ann(ann)

            config = {
                "main_config": main_config,
                "analyses_config": analyses_config,
                "device_config": device_config,
            }

            process.analyze_dataset(
                conn=conn, script_params=script_params, dataset=dataset, config=config
            )

        logger.info(f"End time: {datetime.now()}")

    finally:
        logger.info("Closing connection")

        # Outputting log for user
        client.setOutput("Message", rstring(log_string.getvalue()))

        client.closeSession()


if __name__ == "__main__":
    run_script()
    # run_script_local()
