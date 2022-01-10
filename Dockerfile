# syntax=docker/dockerfile:1
FROM python:3.9.5-slim-buster

# Set work directory. This instructs Docker to use this path as the default 
# location for all subsequent commands. By doing this, we do not have to type 
# out full file paths but can use relative paths based on the working directory.
WORKDIR /src

# set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1

# install python requirements
RUN python -m pip install --upgrade pip
COPY ./requirements.txt /src/requirements.txt
RUN python -m pip install -r requirements.txt

# copy project to the work directory
COPY . /src