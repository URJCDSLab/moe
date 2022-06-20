FROM python:3.9.6
RUN apt-get update

RUN ["mkdir", "/setup"]
COPY requirements.txt /setup/requirements.txt
RUN pip  install -r /setup/requirements.txt

VOLUME app
WORKDIR /app
CMD python setup.py install && python -m src.results.experiments