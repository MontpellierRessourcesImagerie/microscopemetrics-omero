FROM gitlab-registry.in2p3.fr/fbi-data/dockers-projects/omero-server/omero-server:latest
LABEL authors="Julio Mateos Langerak"


RUN mkdir $OMERODIR/lib/scripts/omero/microscope_metrics
COPY microscope-metrics/* $OMERODIR/lib/scripts/omero/microscope_metrics/
COPY microscopemetrics_omero/main_config.yaml /etc/omero_metrics/
USER root
RUN chmod -R 755 $OMERODIR/lib/scripts/omero/microscope_metrics/
COPY microscopemetrics-0.2.2-py3-none-any.whl .
COPY microscopemetrics_omero-0.2.0-py3-none-any.whl .
RUN /opt/omero/server/venv3/bin/pip install -U microscopemetrics-0.2.2-py3-none-any.whl  # remove
RUN /opt/omero/server/venv3/bin/pip install -U microscopemetrics_omero-0.2.0-py3-none-any.whl  # remove
#RUN /opt/omero/server/venv3/bin/pip install -U  \
#    microscopemetrics==0.2.2  \
#    microscopemetrics-omero

USER omero-server
