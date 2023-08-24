FROM openmicroscopy/omero-server:latest
LABEL authors="Julio Mateos Langerak"

WORKDIR /opt/omero/server/OMERO.server
ENV VIRTUAL_ENV=/opt/omero/server/venv3
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN mkdir -p lib/scripts/omero/microscope_metrics
COPY src/omero_scripts/* lib/scripts/omero/microscope_metrics/
COPY dist/microscopemetrics_omero-0.1.0-py3-none-any.whl .
USER root
RUN yum install -y git
#RUN curl -sSL https://install.python-poetry.org | python3 -
#RUN pip install poetry
RUN pip install --upgrade pip
#RUN yum update -y
#RUN yum install -y git
RUN pip install microscopemetrics
RUN pip install microscopemetrics_omero-0.1.0-py3-none-any.whl

USER omero-server

RUN omero admin restart
