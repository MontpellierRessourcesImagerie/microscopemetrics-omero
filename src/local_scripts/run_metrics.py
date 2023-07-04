
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
            process.process_dataset(
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

