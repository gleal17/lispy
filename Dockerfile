FROM python:3.6

ENV PYTHONUNBUFFERED 1
RUN mkdir -p /code
WORKDIR /code
ADD requirements.txt /code/
RUN python3 -m pip install -r requirements.txt
ADD . /code/
EXPOSE 8000