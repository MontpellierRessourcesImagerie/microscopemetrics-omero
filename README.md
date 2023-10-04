![PyPI - Version](https://img.shields.io/pypi/v/microscopemetrics_omero)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/microscopemetrics_omero)

[//]: # (![GitHub Workflow Status &#40;with event&#41;]&#40;https://img.shields.io/github/actions/workflow/status/MontpellierRessourcesImagerie/microscope-metrics/run_tests_push.yml&#41;)
[![GPLv2 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)


# microscopemetrics-omero
_microscopemetrics-omero_ is a python package for using
[OMERO](https://www.openmicroscopy.org/omero/) as a data management solution for 
[microscope-metrics](https://github.com/MontpellierRessourcesImagerie/microscope-metrics)

microscopemetrcis-omero provides a set of functions to store and retrieve data from OMERO as defined
by the [microscopemetrics-schema](https://github.com/MontpellierRessourcesImagerie/microscopemetrics-schema)

It also provides a number of scripts to run microscope-metrics either in OMERO's scripting service
or from outside of the OMERO server but using OMERO as a data management solution.

## Documentation

Documentation is still scarce. You may find it at [Documentation](./docs) pages.

For the time being please refer to the [tests](./tests)
directory to find some example code

## Installation

Install microscopemetrics-omero with pip

```bash
  pip install microscopemetrics-omero
```

You may want to do that in the virtual environment of your OMERO instance and then install the scripts in
[./tests/omero-server/microscope_metrics](./tests/omero-server/microscope_metrics) 
in the OMERO server's script directory.


For development, we use [poetry](https://python-poetry.org/)
After [installing poetry](https://python-poetry.org/docs/#installation), you can install microscope-metrics running the following command 
in the root directory of the project

```bash
  poetry install
```

## Running Tests

To run tests, use pytest from the root directory of the project

```bash
  pytest 
```

