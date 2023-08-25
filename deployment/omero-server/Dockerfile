FROM gitlab-registry.in2p3.fr/fbi-data/dockers-projects/omero-server/omero-server:latest
LABEL authors="Julio Mateos Langerak"


RUN mkdir $OMERODIR/lib/scripts/omero/microscope_metrics
COPY microscope-metrics/* $OMERODIR/lib/scripts/omero/microscope_metrics/
USER root
RUN chmod -R 755 $OMERODIR/lib/scripts/omero/microscope_metrics/
RUN /opt/omero/server/venv3/bin/pip install -U  \
    microscopemetrics==0.2.2  \
    microscopemetrics-omero

USER omero-server
