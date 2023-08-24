FROM openmicroscopy/omero-server:latest
LABEL authors="Julio Mateos Langerak"

WORKDIR /opt/omero/server/OMERO.server
ENV VIRTUAL_ENV=/opt/omero/server/venv3
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt requirements.txt
COPY microscope-metrics lib/scripts/omero/
USER root
RUN yum install -y git
USER omero-server
#RUN curl -sSL https://install.python-poetry.org | python3 -
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN omero admin restart
