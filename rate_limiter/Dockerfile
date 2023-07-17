FROM python:3.9-slim AS base

# Install dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends \ 
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv venv

# install the requirements into a build, to avoid reinstallation
FROM base as build-venv
RUN /venv/bin/pip3 install RPi.GPIO --no-cache-dir --index-url https://www.piwheels.org/simple

# Copy files into the build image
FROM build-venv as build
COPY . /app
WORKDIR /app
CMD ["/venv/bin/python3", "rate_limit.py"]